#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <string>

// Function to load data from a file
bool loadData(const std::string& filename, int dims, std::vector<std::vector<double>>& data_points);

// Function to calculate the Euclidean distance between two points
double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2);

// Assign points to the nearest centroid
std::vector<int> assignPointsToClusters(const std::vector<std::vector<double>>& data_points, const std::vector<std::vector<double>>& centroids);

// Function to initialize centroids randomly from the data points
void initializeCentroids(const std::vector<std::vector<double>>& data_points, int num_clusters, int seed, std::vector<std::vector<double>>& centroids);

// Function to run the sequential version of K-Means
std::pair<int, double> runKMeansSequential(const std::vector<std::vector<double>>& data_points, std::vector<std::vector<double>>& centroids, int max_iters, double threshold);

// Function to run the CUDA version of K-Means
std::pair<int, double> runKMeansCUDA(const std::vector<std::vector<double>>& data_points, std::vector<std::vector<double>>& centroids, int max_iters, double threshold);

// Function to run the Thrust version of K-Means
std::pair<int, double> runKMeansThrust(const std::vector<std::vector<double>>& data_points, std::vector<std::vector<double>>& centroids, int max_iters, double threshold);

// Function to run the Thrust version of K-Means
std::pair<int, double> runKMeansSharedMemoryCUDA(const std::vector<std::vector<double>>& data_points, std::vector<std::vector<double>>& centroids, int max_iters, double threshold);

#endif // KMEANS_H
