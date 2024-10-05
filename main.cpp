#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "kmeans.h"

void printUsage() {
    std::cout << "Usage: ./kmeans -k num_clusters -d dims -i input_file -m max_iters -t threshold -s seed [-c] [--use_cpu | --use_cuda | --use_thrust | --use_shmem]\n";
}

int main(int argc, char *argv[]) {
    // Parsing command line arguments

    // -k num_cluster   : an integer specifying the number of clusters
    // -d dims          : an integer specifying the dimension of the points
    // -i inputfilename : a string specifying the input filename
    // -m max_num_iter  : an integer specifying the maximum number of iterations
    // -t threshold     : a double specifying the threshold for convergence test.
    // -c               : a flag to control the output of your program. If -c is specified, 
    //                     your program should output the centroids of all clusters. 
    //                     If -c is not specified, your program should output the labels of all points. See details below.
    // -s seed          : an integer specifying the seed for rand(). This is used by the autograder 
    //                     to simplify the correctness checking process.
    
    int num_clusters = 0;
    int dims = 0;
    std::string input_file;
    int max_iters = 150;
    double threshold = 0.00001;
    int seed = 0;
    bool output_centroids = false;
    bool use_cpu = false, use_cuda = false, use_thrust = false, use_shmem = false;

    if (argc < 13) {
        printUsage();
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-k" && i + 1 < argc) {
            num_clusters = std::atoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            dims = std::atoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            max_iters = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (arg == "-c") {
            output_centroids = true;
        } else if (arg == "--use_cpu") {
            use_cpu = true;
        } else if (arg == "--use_cuda") {
            use_cuda = true;
        } else if (arg == "--use_thrust") {
            use_thrust = true;
        } else if (arg == "--use_shmem") {
            use_shmem = true;
        } else {
            printUsage();
            return 1;
        }
    }

    // Load data from input file
    std::vector<std::vector<double>> data_points;
    if (!loadData(input_file, dims, data_points)) {
        std::cerr << "Failed to load data from input file: " << input_file << std::endl;
        return 1;
    }

    // Initialize centroids randomly
    //TODO: does this initialization need parallelism?
    std::vector<std::vector<double>> centroids;
    initializeCentroids(data_points, num_clusters, seed, centroids);

    // Run K-Means (choose implementation based on the command line flag)
    // int iterations = 0;
    // auto start = std::chrono::high_resolution_clock::now();


    // Run K-Means (choose implementation based on the command line flag)
    std::pair<int, double> results;
    if (use_cpu) {
        results = runKMeansSequential(data_points, centroids, max_iters, threshold);
    } else if (use_cuda) {
        results = runKMeansCUDA(data_points, centroids, max_iters, threshold);
    // } else if (use_thrust) {
    //     iterations = runKMeansThrust(data_points, centroids, max_iters, threshold);
    } else if (use_shmem) {
        results = runKMeansSharedMemoryCUDA(data_points, centroids, max_iters, threshold);
    } else {
        std::cerr << "Error: No implementation type specified. Use --use_cpu, --use_cuda, --use_thrust, or --use_shmem." << std::endl;
        return 1;
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // double time_per_iter = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)iterations;

    // Output number of iterations and time per iterations
    int iterations = results.first;
    double total_time = results.second;
    double time_per_iter = total_time / (double)iterations;
    
    // Output results
    // std::cout << iterations << "," << time_per_iter << std::endl;
    printf("%d,%lf\n", iterations, time_per_iter);
    //TODO: change to this form
    //printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms)

    // Output either centroids or cluster assignments based on -c flag
    if (output_centroids) {
        // Print centroids
        for (int clusterId = 0; clusterId < num_clusters; ++clusterId) {
            printf("%d ", clusterId);
            for (int d = 0; d < dims; ++d) {
                printf("%lf ", centroids[clusterId][d]);
            }
            printf("\n");
        }
    } else {
        // Print cluster assignments
        //TODO: is this form ok? 2.1 Points Labels
        //printf("clusters:")
        //for (int p=0; p < nPoints; p++)
        //printf(" %d", clusterId_of_point[p]);
        std::vector<int> cluster_assignments = assignPointsToClusters(data_points, centroids);
        printf("clusters:");
        for (int p = 0; p < cluster_assignments.size(); ++p) {
            printf(" %d", cluster_assignments[p]);
        }
        printf("\n");
    }

    return 0;
}
