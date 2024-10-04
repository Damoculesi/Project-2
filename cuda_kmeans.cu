#include "kmeans.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <cfloat>
#include <chrono>


__global__ void assignPointsToCentroids(const double* data_points, const double* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    double min_distance = DBL_MAX;
    int closest_cluster = -1;

    for (int c = 0; c < num_clusters; ++c) {
        double distance = 0.0;
        for (int d = 0; d < dims; ++d) {
            double diff = data_points[point_idx * dims + d] - centroids[c * dims + d];
            distance += diff * diff;
        }
        if (distance < min_distance) {
            min_distance = distance;
            closest_cluster = c;
        }
    }

    cluster_assignments[point_idx] = closest_cluster;
}

__global__ void updateCentroids(const double* data_points, const int* cluster_assignments, double* new_centroids, int* points_per_cluster, int num_points, int num_clusters, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    int cluster_id = cluster_assignments[point_idx];

    for (int d = 0; d < dims; ++d) {
        atomicAdd(&new_centroids[cluster_id * dims + d], data_points[point_idx * dims + d]);
    }

    if (threadIdx.x == 0) {
        atomicAdd(&points_per_cluster[cluster_id], 1);
    }
}


int runKMeansCUDA(const std::vector<std::vector<double>>& data_points,
                  std::vector<std::vector<double>>& centroids,
                  int max_iters, double threshold) {

    int num_points = data_points.size();
    int num_clusters = centroids.size();
    int dims = centroids[0].size();

    // Flatten data_points and centroids for GPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
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
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "Host preparation time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

    // Allocate memory on GPU
    start_cpu = std::chrono::high_resolution_clock::now();
    double* d_data_points;
    double* d_centroids;
    int* d_cluster_assignments;
    cudaMalloc(&d_data_points, num_points * dims * sizeof(double));
    cudaMalloc(&d_centroids, num_clusters * dims * sizeof(double));
    cudaMalloc(&d_cluster_assignments, num_points * sizeof(int));
    end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU memory allocation time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

    // Copy data from host to device
    start_cpu = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data_points, flat_data_points.data(), num_points * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flat_centroids.data(), num_clusters * dims * sizeof(double), cudaMemcpyHostToDevice);
    end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "Data transfer from host to device time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

    // Time measurement using CUDA events for kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    // Main loop for K-Means
    int iterations = 0;
    while (iterations < max_iters) {
        // Step 1: Assign points to nearest centroid
        auto kernel_start = std::chrono::high_resolution_clock::now();
        assignPointsToCentroids<<<grid_size, block_size>>>(d_data_points, d_centroids, d_cluster_assignments, num_points, num_clusters, dims);
        cudaDeviceSynchronize();
        auto kernel_end = std::chrono::high_resolution_clock::now();
        std::cout << "Kernel execution time for assign points (iteration " << iterations << "): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count() << " ms" << std::endl;

        // Step 2: Update centroids
        // Allocate memory for new centroids and points count
        start_cpu = std::chrono::high_resolution_clock::now();
        double* d_new_centroids;
        int* d_points_per_cluster;
        cudaMalloc(&d_new_centroids, num_clusters * dims * sizeof(double));
        cudaMalloc(&d_points_per_cluster, num_clusters * sizeof(int));
        cudaMemset(d_new_centroids, 0, num_clusters * dims * sizeof(double));
        cudaMemset(d_points_per_cluster, 0, num_clusters * sizeof(int));
        end_cpu = std::chrono::high_resolution_clock::now();
        std::cout << "GPU memory allocation and initialization for new centroids (iteration " << iterations << "): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

        // Kernel to update centroids
        kernel_start = std::chrono::high_resolution_clock::now();
        updateCentroids<<<grid_size, block_size>>>(d_data_points, d_cluster_assignments, d_new_centroids, d_points_per_cluster, num_points, num_clusters, dims);
        cudaDeviceSynchronize();
        kernel_end = std::chrono::high_resolution_clock::now();
        std::cout << "Kernel execution time for update centroids (iteration " << iterations << "): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count() << " ms" << std::endl;

        // Copy new centroids back to host
        start_cpu = std::chrono::high_resolution_clock::now();
        std::vector<double> new_flat_centroids(num_clusters * dims);
        std::vector<int> points_per_cluster(num_clusters);
        cudaMemcpy(new_flat_centroids.data(), d_new_centroids, num_clusters * dims * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(points_per_cluster.data(), d_points_per_cluster, num_clusters * sizeof(int), cudaMemcpyDeviceToHost);
        end_cpu = std::chrono::high_resolution_clock::now();
        std::cout << "Data transfer from device to host time (iteration " << iterations << "): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

        // Calculate new centroids by averaging
        start_cpu = std::chrono::high_resolution_clock::now();
        bool converged = true;
        for (int c = 0; c < num_clusters; ++c) {
            for (int d = 0; d < dims; ++d) {
                if (points_per_cluster[c] > 0) {
                    new_flat_centroids[c * dims + d] /= points_per_cluster[c];
                }
                double diff = fabs(new_flat_centroids[c * dims + d] - flat_centroids[c * dims + d]);
                if (diff > threshold) {
                    converged = false;
                }
                flat_centroids[c * dims + d] = new_flat_centroids[c * dims + d];
            }
        }
        end_cpu = std::chrono::high_resolution_clock::now();
        std::cout << "Host-side centroid calculation time (iteration " << iterations << "): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

        // Check convergence
        if (converged) {
            break;
        }

        iterations++;

        // Free temporary memory for centroids
        cudaFree(d_new_centroids);
        cudaFree(d_points_per_cluster);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total GPU kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy final centroids back to the host vector
    start_cpu = std::chrono::high_resolution_clock::now();
    cudaMemcpy(flat_centroids.data(), d_centroids, num_clusters * dims * sizeof(double), cudaMemcpyHostToDevice);
    for (int c = 0; c < num_clusters; ++c) {
        for (int d = 0; d < dims; ++d) {
            centroids[c][d] = flat_centroids[c * dims + d];
        }
    }
    end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "Final centroid copy back to host time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() << " ms" << std::endl;

    // Free GPU memory
    cudaFree(d_data_points);
    cudaFree(d_centroids);
    cudaFree(d_cluster_assignments);

    return iterations;
}
