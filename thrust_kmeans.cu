#include "kmeans.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <cmath>
#include <cfloat>           // For DBL_MAX
#include <cuda_runtime.h>   // For atomicAdd
#include <iostream>
#include <chrono>

// Functor to assign points to the nearest centroid
struct NearestCentroidFunctor {
    const double* data_points;
    const double* centroids;
    int num_centroids;
    int dims;

    NearestCentroidFunctor(const double* _data_points, const double* _centroids, int _num_centroids, int _dims)
        : data_points(_data_points), centroids(_centroids), num_centroids(_num_centroids), dims(_dims) {}

    __host__ __device__
    int operator()(int idx) const {
        const double* point = data_points + idx * dims;
        double min_distance = DBL_MAX;
        int closest_centroid = -1;

        for (int c = 0; c < num_centroids; ++c) {
            const double* centroid = centroids + c * dims;
            double distance = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = point[d] - centroid[d];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = c;
            }
        }
        return closest_centroid;
    }
};

// Functor to update centroids using atomic operations
struct CentroidUpdateFunctor {
    const double* data_points;
    const int* cluster_assignments;
    double* centroid_sums;
    int* points_per_centroid;
    int dims;

    CentroidUpdateFunctor(const double* _data_points, const int* _cluster_assignments, double* _centroid_sums, int* _points_per_centroid, int _dims)
        : data_points(_data_points), cluster_assignments(_cluster_assignments), centroid_sums(_centroid_sums), points_per_centroid(_points_per_centroid), dims(_dims) {}

    __device__  // Ensure this functor is compiled for device
    void operator()(int idx) const {
        int cluster = cluster_assignments[idx];
        const double* point = data_points + idx * dims;
        for (int d = 0; d < dims; ++d) {
            // Use atomicAdd for double
            atomicAdd(&centroid_sums[cluster * dims + d], point[d]);
        }
        atomicAdd(&points_per_centroid[cluster], 1);
    }
};

// Functor to compute the difference between old and new centroids
struct CentroidDiffFunctor {
    const double* old_centroids;
    const double* new_centroids;
    int dims;

    CentroidDiffFunctor(const double* _old_centroids, const double* _new_centroids, int _dims)
        : old_centroids(_old_centroids), new_centroids(_new_centroids), dims(_dims) {}

    __host__ __device__
    double operator()(int idx) const {
        const double* old_centroid = old_centroids + idx * dims;
        const double* new_centroid = new_centroids + idx * dims;
        double dist = 0.0;
        for (int d = 0; d < dims; ++d) {
            double diff = old_centroid[d] - new_centroid[d];
            dist += diff * diff;
        }
        return sqrt(dist);
    }
};

std::pair<int, double> runKMeansThrust(const std::vector<std::vector<double>>& data_points,
                                       std::vector<std::vector<double>>& centroids,
                                       int max_iters, double threshold) {
    int num_points = data_points.size();
    int num_centroids = centroids.size();
    int dims = data_points[0].size();

    // Flatten data_points and centroids
    std::vector<double> h_data_points_flat(num_points * dims);
    for (int i = 0; i < num_points; ++i) {
        std::copy(data_points[i].begin(), data_points[i].end(), h_data_points_flat.begin() + i * dims);
    }

    std::vector<double> h_centroids_flat(num_centroids * dims);
    for (int i = 0; i < num_centroids; ++i) {
        std::copy(centroids[i].begin(), centroids[i].end(), h_centroids_flat.begin() + i * dims);
    }

    // Transfer data to device
    thrust::device_vector<double> d_data_points = h_data_points_flat;
    thrust::device_vector<double> d_centroids = h_centroids_flat;
    thrust::device_vector<int> d_cluster_assignments(num_points);

    thrust::device_vector<double> d_old_centroids(num_centroids * dims);

    int iterations = 0;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    while (iterations < max_iters) {
        // Step 1: Assign points to the nearest centroid
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_points),
            d_cluster_assignments.begin(),
            NearestCentroidFunctor(
                thrust::raw_pointer_cast(d_data_points.data()),
                thrust::raw_pointer_cast(d_centroids.data()),
                num_centroids, dims));

        // Step 2: Save current centroids for convergence check
        d_old_centroids = d_centroids;

        // Step 3: Update centroids based on cluster assignments
        // Initialize centroid sums and counts
        thrust::device_vector<double> d_centroid_sums(num_centroids * dims, 0.0);
        thrust::device_vector<int> d_points_per_centroid(num_centroids, 0);

        // Raw pointers for device access
        double* centroid_sums = thrust::raw_pointer_cast(d_centroid_sums.data());
        int* points_per_centroid = thrust::raw_pointer_cast(d_points_per_centroid.data());
        const double* data_points = thrust::raw_pointer_cast(d_data_points.data());
        const int* cluster_assignments = thrust::raw_pointer_cast(d_cluster_assignments.data());

        // Update centroid sums and counts using a functor
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_points),
            CentroidUpdateFunctor(
                data_points,
                cluster_assignments,
                centroid_sums,
                points_per_centroid,
                dims));

        // Compute new centroids by dividing sums by counts
        // Raw pointer for old centroids
        const double* old_centroids = thrust::raw_pointer_cast(d_old_centroids.data());

        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_centroids * dims),
            d_centroids.begin(),
            [=] __device__ (int idx) -> double {
                int c = idx / dims;
                int count = points_per_centroid[c];
                if (count > 0) {
                    return centroid_sums[idx] / count;
                } else {
                    return old_centroids[idx];  // Keep old centroid if no points assigned
                }
            });

        // Step 4: Check for convergence
        double max_change = thrust::transform_reduce(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_centroids),
            CentroidDiffFunctor(
                thrust::raw_pointer_cast(d_old_centroids.data()),
                thrust::raw_pointer_cast(d_centroids.data()),
                dims),
            0.0,
            thrust::maximum<double>());

        if (max_change <= threshold) {
            break;
        }

        iterations++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Copy final centroids back to host
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids_flat.begin());

    for (int k = 0; k < num_centroids; ++k) {
        std::copy(
            h_centroids_flat.begin() + k * dims,
            h_centroids_flat.begin() + (k + 1) * dims,
            centroids[k].begin());
    }

    return std::make_pair(iterations, total_time);
}
