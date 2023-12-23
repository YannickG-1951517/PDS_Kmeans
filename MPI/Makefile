all: kmeans

clean:
	rm -f kmeans

kmeans: main_startcode.cpp rng.cpp
	mpicxx -O2 -o kmeans_mpi main_startcode.cpp rng.cpp
