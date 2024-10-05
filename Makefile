CUDA_ARCH := -arch=sm_75
CXXFLAGS := -O2 -std=c++14 --expt-extended-lambda

# Default target
all: kmeans

# Link object files with nvcc
kmeans: main.o seq_kmeans.o utils.o thrust_kmeans.o cuda_kmeans.o shmem_kmeans.o
	nvcc $(CXXFLAGS) $(CUDA_ARCH) main.o seq_kmeans.o utils.o thrust_kmeans.o cuda_kmeans.o shmem_kmeans.o -o kmeans

# Compile host code with g++
main.o: main.cpp
	g++ $(CXXFLAGS) -c main.cpp -o main.o

seq_kmeans.o: seq_kmeans.cpp
	g++ $(CXXFLAGS) -c seq_kmeans.cpp -o seq_kmeans.o

utils.o: utils.cpp
	g++ $(CXXFLAGS) -c utils.cpp -o utils.o

# Compile CUDA code with nvcc
cuda_kmeans.o: cuda_kmeans.cu
	nvcc $(CXXFLAGS) $(CUDA_ARCH) -c cuda_kmeans.cu -o cuda_kmeans.o

shmem_kmeans.o: shmem_kmeans.cu
	nvcc $(CXXFLAGS) $(CUDA_ARCH) -c shmem_kmeans.cu -o shmem_kmeans.o

thrust_kmeans.o: thrust_kmeans.cu
	nvcc $(CXXFLAGS) $(CUDA_ARCH) -c thrust_kmeans.cu -o thrust_kmeans.o

clean:
	rm -f *.o kmeans
