all: reduction

reduction: reduction.cu
	nvcc -g -O3 -arch=sm_30 $< -o $@

clean:
	rm -rf reduction

