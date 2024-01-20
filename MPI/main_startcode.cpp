#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
// #include "rng.cpp"
#include <mpi.h>

using namespace std;

void usage()
{
    std::cerr << R"XYZ(
Usage:

  kmeans --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--blocks numblocks] [--threads numthreads] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

Arguments:

 --input:

   Specifies input CSV file, number of rows represents number of points, the
   number of columns is the dimension of each point.

 --output:

   Output CSV file, just a single row, with as many entries as the number of
   points in the input file. Each entry is the index of the cluster to which
   the point belongs. The script 'visualize_clusters.py' can show this final
   clustering.

 --k:

   The number of clusters that should be identified.

 --repetitions:

   The number of times the k-means algorithm is repeated; the best clustering
   is kept.

 --blocks:

   Only relevant in CUDA version, specifies the number of blocks that can be
   used.

 --threads:

   Not relevant for the serial version. For the OpenMP version, this number
   of threads should be used. For the CUDA version, this is the number of
   threads per block. For the MPI executable, this should be ignored, but
   the wrapper script 'mpiwrapper.sh' can inspect this to run 'mpirun' with
   the correct number of processes.

 --seed:

   Specifies a seed for the random number generator, to be able to get
   reproducible results.

 --trace:

   Debug option - do NOT use this when timing your program!

   For each repetition, the k-means algorithm goes through a sequence of
   increasingly better cluster assignments. If this option is specified, this
   sequence of cluster assignments should be written to a CSV file, similar
   to the '--output' option. Instead of only having one line, there will be
   as many lines as steps in this sequence. If multiple repetitions are
   specified, only the results of the first repetition should be logged
   for clarity. The 'visualize_clusters.py' program can help to visualize
   the data logged in this file.

 --centroidtrace:

   Debug option - do NOT use this when timing your program!

   Should also only log data during the first repetition. The resulting CSV
   file first logs the randomly chosen centroids from the input data, and for
   each step in the sequence, the updated centroids are logged. The program
   'visualize_centroids.py' can be used to visualize how the centroids change.

)XYZ";
    exit(-1);
}

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream &input, std::vector<double> &allData, int &numRows, int &numCols)
{
    if (!input.is_open())
        throw std::runtime_error("Input file is not open");

    allData.resize(0);
    numRows = 0;
    numCols = -1;

    CSVReader inReader(input);
    int numColsExpected = -1;
    int line = 1;
    std::vector<double> row;

    while (inReader.read(row))
    {
        if (numColsExpected == -1)
        {
            numColsExpected = row.size();
            if (numColsExpected <= 0)
                throw std::runtime_error("Unexpected error: 0 columns");
        }
        else if (numColsExpected != (int)row.size())
            throw std::runtime_error("Incompatible number of colums read in line " + std::to_string(line) + ": expecting " + std::to_string(numColsExpected) + " but got " + std::to_string(row.size()));

        for (auto x : row)
            allData.push_back(x);

        line++;
    }

    numRows = (size_t)allData.size()/numColsExpected;
    numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string &n, int rank)
{
    FileCSVWriter f;
    if (rank != 0)
        return f;
    if (n.length() != 0)
    {
        f.open(n);
        if (!f.is_open())
            std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
    }
    return f;
}

int kmeans(Rng &rng, const std::string &inputFile, const std::string &outputFileName,
           int numClusters, int repetitions, int numBlocks, int numThreads,
           const std::string &centroidDebugFileName, const std::string &clusterDebugFileName)
{
    // If debug filenames are specified, this opens them. The is_open method
    // can be used to check if they are actually open and should be written to.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Read file and assign data if rank is 0
    vector<double> data;
    int num_rows;
    int num_columns;

    FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName, rank);
    FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName, rank);

    FileCSVWriter csvOutputFile;

    double start;
    if (rank == 0) {
        csvOutputFile.open(outputFileName);
        if (!csvOutputFile.is_open())
        {
            std::cerr << "Unable to open output file " << outputFileName << std::endl;
            return -1;
        }

        ifstream file(inputFile);
        readData(file, data, num_rows, num_columns);

        // start time
        start = MPI_Wtime();
    }

    // broadcast number of rows and columns
    MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // This is a basic timer from std::chrono ; feel free to use the appropriate timer for
    // each of the technologies, e.g. OpenMP has omp_get_wtime()
    // Timer timer;

    double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better
    vector<int> bestClusters(num_rows);
    std::vector<size_t> stepsPerRepetition(repetitions); // to save the number of steps each rep needed

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep
    // the best result of these repetitions.

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int count = ceil(num_rows/(double)size)*num_columns;


    std::vector<int> displacements(size);
    std::vector<int> counts(size);
    int oversize = size*count - num_rows*num_columns;
    for (int i = 0; i < size; i++) {
        displacements[i] = i * count;
        if (i == size - 1) {
            counts[i] = count - oversize;
        }
        else
            counts[i] = count;
    }
    int rcv_count = counts[rank];

    std::vector<double> local_data(rcv_count);
    vector<int> local_clusters(rcv_count/num_columns, -1);

    MPI_Scatterv(data.data(), counts.data(), displacements.data(), MPI_DOUBLE, local_data.data(), rcv_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < size; i++) {
        displacements[i] = i * count/num_columns;
        counts[i] = counts[i]/num_columns;
    }
    rcv_count = counts[rank];

    for (int r = 0 ; r < repetitions ; r++)
    {
        size_t numSteps = 0;

        vector<size_t> indices(numClusters);
        rng.pickRandomIndices(num_rows, indices);

        vector<double> centroids(num_columns*numClusters);
        if (rank == 0) {
            for (int i = 0; i < indices.size(); i++) {
                for (int j = 0; j < num_columns; j++) {
                    centroids[i*num_columns+j] = data[indices[i]*num_columns+j];
                }
            }
        }
        vector<int> clusters(num_rows, -1);

        bool changed = true;
        while (changed) {
            numSteps++;

            changed = false;
            double distanceSquaredSum = 0;


            MPI_Bcast(centroids.data(), numClusters*num_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatterv(clusters.data(), counts.data(), displacements.data(), MPI_INT, local_clusters.data(), rcv_count, MPI_INT, 0, MPI_COMM_WORLD);


            int localNumRows = local_data.size()/num_columns;
            bool localChanged = false;

            for (int i = 0; i < localNumRows; i++) {
                double minDistance = numeric_limits<double>::max(); // can only get better
                int clusterIndex;
                for (int k = 0; k < numClusters; k++) {
                    double currentDistance = 0;
                    for (int j = 0; j < num_columns; j++) {
                        currentDistance += pow((local_data[i * num_columns + j] - centroids[k * num_columns + j]), 2);
                    }
                    if (minDistance > currentDistance) {
                        minDistance = currentDistance;
                        clusterIndex = k;
                    }
                }

                distanceSquaredSum += minDistance;

                if (clusterIndex != local_clusters[i]) {
                    local_clusters[i] = clusterIndex;
                    localChanged = true;
                }
            }

            double totalDistanceSquaredSum = 0;

            MPI_Reduce(&distanceSquaredSum, &totalDistanceSquaredSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_clusters.data(), rcv_count, MPI_INT, clusters.data(), counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);


            MPI_Reduce(&localChanged, &changed, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
            
            MPI_Bcast(&changed, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                if (clustersDebugFile.is_open()) {
                    clustersDebugFile.write(clusters);
                }
                if (centroidDebugFile.is_open()) {
                    for (int i = 0; i < numClusters; i++) {
                        vector<double> centroid(num_columns);
                        for ( int j = 0; j < num_columns; j++) {
                            centroid[j] = centroids[i*num_columns+j];
                        }
                        centroidDebugFile.write(centroid);
                    }
                }
            
                if (changed) {
                    vector<vector<double>> pointsTotals(numClusters, vector<double> (num_columns+1, 0)); // per centroid x, y, #points 
                    for (int i = 0; i < num_rows; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            pointsTotals[clusters[i]][j] += data[i*num_columns+j];
                        }
                        pointsTotals[clusters[i]][num_columns]++;
                    }
                    for (int i = 0; i < numClusters; i++) {
                        for (int j = 0; j < num_columns; j++) {
                            centroids[i * num_columns + j] = pointsTotals[i][j]/pointsTotals[i][num_columns];
                        }
                    }
                }
                if (totalDistanceSquaredSum < bestDistSquaredSum) {
                    bestClusters = clusters;
                    bestDistSquaredSum = totalDistanceSquaredSum;
                }
            }
        }

        stepsPerRepetition[r] = numSteps;

        // Make sure debug logging is only done on first iteration ; subsequent checks
        // with is_open will indicate that no logging needs to be done anymore.
        if (rank == 0 && centroidDebugFile.is_open())
        centroidDebugFile.close();
        if (rank == 0 && clustersDebugFile.is_open())
        clustersDebugFile.close();
    }

    // timer.stop();
    if (rank == 0) {
        double time = MPI_Wtime() - start;

        // Some example output, of course you can log your timing data anyway you like.
        std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
        std::cout << "MPI," << numBlocks << "," << numThreads << "," << inputFile << ","
                << rng.getUsedSeed() << "," << numClusters << ","
                << repetitions << "," << bestDistSquaredSum << "," << time
                << std::endl;

        // Write the number of steps per repetition, kind of a signature of the work involved
        if (csvOutputFile.is_open()) {
            csvOutputFile.write(stepsPerRepetition, "# Steps: ");
            // Write best clusters to csvOutputFile, something like
            csvOutputFile.write(bestClusters);
            csvOutputFile.close();
        }
    }
    return 0;
}

int mainCxx(const std::vector<std::string> &args)
{
    if (args.size()%2 != 0)
        usage();

    std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
    unsigned long seed = 0;

    int numClusters = -1, repetitions = -1;
    int numBlocks = 1, numThreads = 1;
    for (int i = 0 ; i < args.size() ; i += 2)
    {
        if (args[i] == "--input")
            inputFileName = args[i+1];
        else if (args[i] == "--output")
            outputFileName = args[i+1];
        else if (args[i] == "--centroidtrace")
            centroidTraceFileName = args[i+1];
        else if (args[i] == "--trace")
            clusterTraceFileName = args[i+1];
        else if (args[i] == "--k")
            numClusters = stoi(args[i+1]);
        else if (args[i] == "--repetitions")
            repetitions = stoi(args[i+1]);
        else if (args[i] == "--seed")
            seed = stoul(args[i+1]);
        else if (args[i] == "--blocks")
            numBlocks = stoi(args[i+1]);
        else if (args[i] == "--threads")
            numThreads = stoi(args[i+1]);
        else
        {
            std::cerr << "Unknown argument '" << args[i] << "'" << std::endl;
            return -1;
        }
    }

    if (inputFileName.length() == 0 || outputFileName.length() == 0 || numClusters < 1 || repetitions < 1 || seed == 0)
        usage();

    Rng rng(seed);

    return kmeans(rng, inputFileName, outputFileName, numClusters, repetitions,
                  numBlocks, numThreads, centroidTraceFileName, clusterTraceFileName);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    std::vector<std::string> args;
    for (int i = 1 ; i < argc ; i++)
        args.push_back(argv[i]);

    int returnValue = mainCxx(args);

    MPI_Finalize();

    return returnValue;
}
