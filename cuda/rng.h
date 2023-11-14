#pragma once

#include <random>
#include <vector>

class Rng
{
public:
    Rng(unsigned long seed);
    ~Rng();

	// Fills 'indices' (so should already have a specific size) with
	// numbers between 0 and n-1 . This can be used in the k-means
	// algorithm to pick the initial points: if there are n points
	// and c initial centroids need to be chosen, create a vector
	// with c entries, and call this function. The vector will then
	// contain the random indices of the points to use.
    void pickRandomIndices(size_t n, std::vector<size_t> &indices);

	unsigned long getUsedSeed() const { return m_seed; }
private:
    std::mt19937 m_rng;
	unsigned long m_seed;
};
