#include "kmeans.h"
#include <random>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

// Function to calculate the Euclidean distance between two points
double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to assign points to the nearest centroid
std::vector<int> assignPointsToClusters(const std::vector<std::vector<double>>& data_points, const std::vector<std::vector<double>>& centroids) {
    std::vector<int> cluster_assignments(data_points.size());

    for (size_t i = 0; i < data_points.size(); ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int closest_cluster = -1;

        // Find the closest centroid to the current point
        for (size_t j = 0; j < centroids.size(); ++j) {
            double distance = calculateDistance(data_points[i], centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = j;
            }
        }

        // Assign the point to the closest centroid
        cluster_assignments[i] = closest_cluster;
    }

    return cluster_assignments;
}

void initializeCentroids(const std::vector<std::vector<double>>& data_points, int num_clusters, int seed, std::vector<std::vector<double>>& centroids) {
    // Set random seed for reproducibility using kmeans_srand
    kmeans_srand(seed);

    // Clear centroids vector just in case
    centroids.clear();

    // Keep track of which indices have already been chosen to avoid duplicates
    std::unordered_set<int> chosen_indices;

    // Select num_clusters unique points from data_points as the initial centroids
    while (centroids.size() < num_clusters) {
        int index = kmeans_rand() % data_points.size();
        
        // Ensure that the same point is not chosen multiple times
        if (chosen_indices.find(index) == chosen_indices.end()) {
            centroids.push_back(data_points[index]);
            chosen_indices.insert(index);
        }
    }
}



bool loadData(const std::string& filename, int dims, std::vector<std::vector<double>>& data_points) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;

    // Read the first line to get the number of points (and skip it)
    if (!std::getline(infile, line)) {
        std::cerr << "Error: Could not read the number of points from the file." << std::endl;
        return false;
    }
    
    int num_points = std::stoi(line);  // Reading the number of points from the first line

    // Read each line corresponding to the data points
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int point_id;  // To read and ignore the point ID
        iss >> point_id;  // Skip the point ID
        
        std::vector<double> point(dims);
        for (int i = 0; i < dims; ++i) {
            if (!(iss >> point[i])) {
                std::cerr << "Error: Incorrect number of dimensions or invalid data in file." << std::endl;
                return false;
            }
        }
        data_points.push_back(point);
    }

    infile.close();
    return true;
}
