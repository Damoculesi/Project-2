#include "kmeans.h"
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

// Function to update centroids based on the average of the assigned points
void updateCentroids(const std::vector<std::vector<double>>& data_points,
                     const std::vector<int>& cluster_assignments,
                     std::vector<std::vector<double>>& centroids,
                     int num_clusters, int dims) {
    std::vector<std::vector<double>> new_centroids(num_clusters, std::vector<double>(dims, 0.0));
    std::vector<int> points_per_cluster(num_clusters, 0);

    // Sum up points for each cluster
    for (size_t i = 0; i < data_points.size(); ++i) {
        int cluster_id = cluster_assignments[i];
        points_per_cluster[cluster_id]++;
        for (int d = 0; d < dims; ++d) {
            new_centroids[cluster_id][d] += data_points[i][d];
        }
    }

    // Calculate the average for each cluster
    for (int k = 0; k < num_clusters; ++k) {
        if (points_per_cluster[k] > 0) {
            for (int d = 0; d < dims; ++d) {
                new_centroids[k][d] /= points_per_cluster[k];
            }
        }
    }

    // Update the centroids
    centroids = new_centroids;
}

// Function to check if centroids have converged
bool checkConvergence(const std::vector<std::vector<double>>& old_centroids,
                      const std::vector<std::vector<double>>& new_centroids,
                      double threshold) {
    for (size_t i = 0; i < old_centroids.size(); ++i) {
        if (calculateDistance(old_centroids[i], new_centroids[i]) > threshold) {
            return false;  // Not converged
        }
    }
    return true;  // Converged
}

// Main function to run K-Means clustering sequentially
std::pair<int, double> runKMeansSequential(const std::vector<std::vector<double>>& data_points,
                        std::vector<std::vector<double>>& centroids,
                        int max_iters, double threshold) {
    int num_clusters = centroids.size();
    int dims = centroids[0].size();
    int num_points = data_points.size();

    std::vector<int> cluster_assignments(num_points);
    int iterations = 0;

    // Start timing the K-Means main loop
    auto start = std::chrono::high_resolution_clock::now();

    // K-Means main loop
    while (iterations < max_iters) {
        // Step 1: Assign points to the nearest centroid
        cluster_assignments = assignPointsToClusters(data_points, centroids);

        // Step 2: Save current centroids for convergence check
        std::vector<std::vector<double>> old_centroids = centroids;

        // Step 3: Update centroids based on cluster assignments
        updateCentroids(data_points, cluster_assignments, centroids, num_clusters, dims);

        // Step 4: Check for convergence
        if (checkConvergence(old_centroids, centroids, threshold)) {
            break;  // Converged
        }

        iterations++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return std::make_pair(iterations, total_time);
}
