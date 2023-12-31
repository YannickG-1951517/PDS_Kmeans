#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
//#include "rng.cpp"

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
void readData(std::ifstream &input, std::vector<double> &allData, size_t &numRows, size_t &numCols)
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

FileCSVWriter openDebugFile(const std::string &n)
{
    FileCSVWriter f;

    if (n.length() != 0)
    {
        f.open(n);
        if (!f.is_open())
            std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
    }
    return f;
}

__global__ void kernel (int numClusters, int num_columns, int num_rows, double data[], double centroids[], int* clusters, bool* changed, double distanceSquaredSum[], int countPerThread) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*countPerThread;
    int last = i + countPerThread;
    while (i < last && i < num_rows) {
        // double minDistance = numeric_limits<double>::max(); // can only get better
        double minDistance = __DBL_MAX__; // can only get better
        int clusterIndex;
        for (int k = 0; k < numClusters; k++) {
            double currentDistance = 0;
            for (int j = 0; j < num_columns; j++) {
                currentDistance += pow((data[i * num_columns + j] - centroids[k * num_columns + j]), 2);
            }
            if (minDistance > currentDistance) {
                minDistance = currentDistance;
                clusterIndex = k;
            }
        }
        distanceSquaredSum[i] = minDistance;

        if (clusterIndex != clusters[i]) {
            clusters[i] = clusterIndex;
            (*changed)= true;
        }
        i++;
    }
}

int kmeans(Rng &rng, const std::string &inputFile, const std::string &outputFileName,
            int numClusters, int repetitions, int numBlocks, int numThreads,
            const std::string &centroidDebugFileName, const std::string &clusterDebugFileName)
{
    // If debug filenames are specified, this opens them. The is_open method
    // can be used to check if they are actually open and should be written to.
    FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
    FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);

    FileCSVWriter csvOutputFile(outputFileName);
    if (!csvOutputFile.is_open())
    {
        std::cerr << "Unable to open output file " << outputFileName << std::endl;
        return -1;
    }
    
    vector<double> data;
    size_t num_rows;
    size_t num_columns;
    ifstream file(inputFile);
    readData(file, data, num_rows, num_columns);


    // This is a basic timer from std::chrono ; feel free to use the appropriate timer for
    // each of the technologies, e.g. OpenMP has omp_get_wtime()
    Timer timer;

    double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better
    vector<int> bestClusters(num_rows);
    std::vector<size_t> stepsPerRepetition(repetitions); // to save the number of steps each rep needed

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep
    // the best result of these repetitions.

    double distanceSquaredSumArray[num_rows];
    double* GPUdistanceSquaredSum;
    cudaMallocManaged(&GPUdistanceSquaredSum, num_rows*sizeof(double));
    bool *GPUchanged;
    cudaMallocManaged(&GPUchanged, sizeof(bool));
    int *GPUclusters;
    cudaMallocManaged(&GPUclusters, num_rows*sizeof(int));
    double *GPUdata;
    cudaMallocManaged(&GPUdata, num_rows*num_columns*sizeof(double));
    double *GPUcentroids;
    cudaMallocManaged(&GPUcentroids, numClusters*num_columns*sizeof(double));
    cudaMemcpy(GPUdata, data.data(), num_rows*num_columns*sizeof(double), cudaMemcpyHostToDevice);

    int countPerThread = num_rows/(numBlocks*numThreads);
    if (num_rows%(numBlocks*numThreads) != 0) {
        countPerThread++;
    }

    // printf("countPerThread: %d\n", countPerThread);

    for (int r = 0 ; r < repetitions ; r++)
    {
        size_t numSteps = 0;

        vector<size_t> indices(numClusters);
        rng.pickRandomIndices(num_rows, indices);


        vector<double> centroids;

        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < num_columns; j++) {
                centroids.push_back(data[indices[i]*num_columns+j]);
            }
        }

        vector<int> clusters(num_rows, -1);
        

        bool changed = true;
        while (changed) {
            numSteps++;

            changed = false;
            double distanceSquaredSum = 0;
            
            cudaMemcpy(GPUchanged, &changed, sizeof(bool), cudaMemcpyHostToDevice);

            cudaMemcpy(GPUclusters, clusters.data(), num_rows*sizeof(int), cudaMemcpyHostToDevice);

            cudaMemcpy(GPUcentroids, centroids.data(), numClusters*num_columns*sizeof(double), cudaMemcpyHostToDevice);

            kernel<<<numBlocks, numThreads>>>(numClusters, num_columns, num_rows, GPUdata, GPUcentroids, GPUclusters, GPUchanged, GPUdistanceSquaredSum, countPerThread);
            // cudaDeviceSynchronize();
            // printf(cudaGetErrorString(cudaGetLastError()));

            cudaMemcpy(&changed, GPUchanged, sizeof(bool), cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_rows; i++) {
                clusters[i] = GPUclusters[i];
                distanceSquaredSum += GPUdistanceSquaredSum[i];
            }


            

            
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

            if (distanceSquaredSum < bestDistSquaredSum) {
                bestClusters = clusters;
                bestDistSquaredSum = distanceSquaredSum;
            }
        }


        stepsPerRepetition[r] = numSteps;

        // Make sure debug logging is only done on first iteration ; subsequent checks
        // with is_open will indicate that no logging needs to be done anymore.
        centroidDebugFile.close();
        clustersDebugFile.close();
    }

    cudaFree(GPUchanged);
    cudaFree(GPUclusters);
    cudaFree(GPUdata);
    cudaFree(GPUcentroids);
    cudaFree(GPUdistanceSquaredSum);

    timer.stop();

    // Some example output, of course you can log your timing data anyway you like.
    std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
    std::cout << "cuda," << numBlocks << "," << numThreads << "," << inputFile << ","
              << rng.getUsedSeed() << "," << numClusters << ","
              << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds()/1e9
              << std::endl;

    // Write the number of steps per repetition, kind of a signature of the work involved
    csvOutputFile.write(stepsPerRepetition, "# Steps: ");
    // Write best clusters to csvOutputFile, something like
    csvOutputFile.write(bestClusters);
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
    std::vector<std::string> args;
    for (int i = 1 ; i < argc ; i++)
        args.push_back(argv[i]);

    return mainCxx(args);
}
