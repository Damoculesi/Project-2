#include "kmeans.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <cfloat>
#include <chrono>

#define MAX_SHARED_MEM 49152  // Maximum shared memory per block (in bytes)

// Kernel to assign data points to the nearest centroid using shared memory
__global__ void assignPointsToCentroidsSharedMem(const double* __restrict__ data_points, const double* __restrict__ centroids, int* cluster_assignments, int num_points, int num_clusters, int dims) {
    extern __shared__ double shared_centroids[];  // Shared memory for centroids

    int tid = threadIdx.x;
    int point_idx = blockIdx.x * blockDim.x + tid;

    if (point_idx >= num_points) return;

    double min_distance = DBL_MAX;
    int closest_cluster = -1;

    // Calculate how many centroids can fit into shared memory
    int max_centroids_in_shared = MAX_SHARED_MEM / (dims * sizeof(double));
    int centroids_per_tile = min(max_centroids_in_shared, num_clusters);

    for (int centroid_start = 0; centroid_start < num_clusters; centroid_start += centroids_per_tile) {
        int tile_size = min(centroids_per_tile, num_clusters - centroid_start);

        // Load centroids into shared memory
        int elements_per_tile = tile_size * dims;
        int threads = blockDim.x;
        int elements_per_thread = (elements_per_tile + threads - 1) / threads;

        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = tid + i * threads;
            if (idx < elements_per_tile) {
                shared_centroids[idx] = centroids[(centroid_start * dims) + idx];
            }
        }
        __syncthreads();

        // Compute distances to centroids in the tile
        for (int c = 0; c < tile_size; ++c) {
            double distance = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = data_points[point_idx * dims + d] - shared_centroids[c * dims + d];
                distance += diff * diff;
            }
            int global_c = centroid_start + c;
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = global_c;
            }
        }
        __syncthreads();
    }

    cluster_assignments[point_idx] = closest_cluster;
}

// Kernel to compute sums and counts for centroid update using global memory atomics
__global__ void updateCentroidsSharedMem(const double* __restrict__ data_points, const int* __restrict__ cluster_assignments, double* centroid_sums, int* points_per_cluster, int num_points, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= num_points) return;

    int cluster_id = cluster_assignments[point_idx];

    // Update centroid sums and counts using atomic operations
    for (int d = 0; d < dims; ++d) {
        atomicAdd(&centroid_sums[cluster_id * dims + d], data_points[point_idx * dims + d]);
    }
    atomicAdd(&points_per_cluster[cluster_id], 1);
}

// Kernel to compute new centroids by averaging and update centroids
__global__ void computeNewCentroidsSharedMem(const double* __restrict__ old_centroids, double* centroids, const double* centroid_sums, const int* points_per_cluster, int num_clusters, int dims) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= num_clusters) return;

    if (points_per_cluster[cluster_id] > 0) {
        for (int d = 0; d < dims; ++d) {
            double new_value = centroid_sums[cluster_id * dims + d] / points_per_cluster[cluster_id];
            centroids[cluster_id * dims + d] = new_value;
        }
    } else {
        // Handle empty clusters by keeping the old centroid
        for (int d = 0; d < dims; ++d) {
            centroids[cluster_id * dims + d] = old_centroids[cluster_id * dims + d];
        }
    }
}

// Main function to run K-Means clustering using CUDA with shared memory
std::pair<int, double> runKMeansSharedMemoryCUDA(const std::vector<std::vector<double>>& data_points,
                                                 std::vector<std::vector<double>>& centroids,
                                                 int max_iters, double threshold) {

    int num_points = data_points.size();
    int num_clusters = centroids.size();
    int dims = centroids[0].size();

    // Flatten data_points and centroids for GPU
    std::vector<double> flat_data_points(num_points * dims);
    std::vector<double> flat_centroids(num_clusters * dims);
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            flat_data_points[i * dims + d] = data_points[i][d];
        }
    }
    for (int c = 0; c < num_clusters; ++c) {
        for (int d = 0; d < dims; ++d) {
            flat_centroids[c * dims + d] = centroids[c][d];
        }
    }

    // Allocate memory on GPU
    double* d_data_points;
    double* d_centroids;
    double* d_old_centroids;  // For convergence check
    int* d_cluster_assignments;
    cudaMalloc(&d_data_points, num_points * dims * sizeof(double));
    cudaMalloc(&d_centroids, num_clusters * dims * sizeof(double));
    cudaMalloc(&d_old_centroids, num_clusters * dims * sizeof(double));
    cudaMalloc(&d_cluster_assignments, num_points * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data_points, flat_data_points.data(), num_points * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flat_centroids.data(), num_clusters * dims * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    int iterations = 0;

    // Allocate memory for centroid sums and point counts
    double* d_centroid_sums;
    int* d_points_per_cluster;
    cudaMalloc(&d_centroid_sums, num_clusters * dims * sizeof(double));
    cudaMalloc(&d_points_per_cluster, num_clusters * sizeof(int));

    // Prepare for convergence check
    std::vector<double> flat_old_centroids = flat_centroids;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (iterations < max_iters) {
        // Step 1: Assign points to nearest centroid using shared memory
        int max_centroids_in_shared = MAX_SHARED_MEM / (dims * sizeof(double));
        int centroids_per_tile = min(max_centroids_in_shared, num_clusters);
        size_t shared_mem_size = centroids_per_tile * dims * sizeof(double);

        assignPointsToCentroidsSharedMem<<<grid_size, block_size, shared_mem_size>>>(d_data_points, d_centroids, d_cluster_assignments, num_points, num_clusters, dims);
        cudaDeviceSynchronize();

        // Step 2: Reset centroid sums and point counts
        cudaMemset(d_centroid_sums, 0, num_clusters * dims * sizeof(double));
        cudaMemset(d_points_per_cluster, 0, num_clusters * sizeof(int));

        // Step 3: Update centroid sums and counts
        updateCentroidsSharedMem<<<grid_size, block_size>>>(d_data_points, d_cluster_assignments, d_centroid_sums, d_points_per_cluster, num_points, dims);
        cudaDeviceSynchronize();

        // Step 4: Copy centroids to d_old_centroids for convergence check
        cudaMemcpy(d_old_centroids, d_centroids, num_clusters * dims * sizeof(double), cudaMemcpyDeviceToDevice);

        // Step 5: Compute new centroids and update centroids on device
        int centroid_block_size = 256;
        int centroid_grid_size = (num_clusters + centroid_block_size - 1) / centroid_block_size;
        computeNewCentroidsSharedMem<<<centroid_grid_size, centroid_block_size>>>(d_old_centroids, d_centroids, d_centroid_sums, d_points_per_cluster, num_clusters, dims);
        cudaDeviceSynchronize();

        // Step 6: Copy centroids back to host to check for convergence
        cudaMemcpy(flat_centroids.data(), d_centroids, num_clusters * dims * sizeof(double), cudaMemcpyDeviceToHost);

        // Step 7: Check for convergence using Euclidean distance
        bool converged = true;
        for (int c = 0; c < num_clusters; ++c) {
            double dist = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = flat_centroids[c * dims + d] - flat_old_centroids[c * dims + d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist > threshold) {
                converged = false;
                break;  // Exit early if any centroid has not converged
            }
        }

        // Prepare for next iteration
        flat_old_centroids = flat_centroids;

        if (converged) {
            break;
        }

        iterations++;
    }

    // End timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    double total_time = static_cast<double>(elapsed_time);

    // Copy final centroids back to host
    for (int c = 0; c < num_clusters; ++c) {
        for (int d = 0; d < dims; ++d) {
            centroids[c][d] = flat_centroids[c * dims + d];
        }
    }

    // Free GPU memory
    cudaFree(d_data_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_cluster_assignments);
    cudaFree(d_centroid_sums);
    cudaFree(d_points_per_cluster);

    return std::make_pair(iterations, total_time);
}
