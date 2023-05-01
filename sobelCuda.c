#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 1024
#define BLOCK_SIZE 32

_global_ void sobel(int* image, int* edge, int rows, int cols) {
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (i > 0 && i < rows-1 && j > 0 && j < cols-1) {
        int gx = -1*image[(i-1)*cols+j-1] + image[(i-1)*cols+j+1]
                 -2*image[i*cols+j-1]     + 2*image[i*cols+j+1]
                 -1*image[(i+1)*cols+j-1] + image[(i+1)*cols+j+1];

        int gy = -1*image[(i-1)*cols+j-1] - 2*image[(i-1)*cols+j] - 1*image[(i-1)*cols+j+1]
                 + image[(i+1)*cols+j-1]  + 2*image[(i+1)*cols+j]  + 1*image[(i+1)*cols+j+1];

        int val = (int)sqrt(gx*gx + gy*gy);
        if (val > 255) {
            val = 255;
        }
        edge[i*cols+j] = val;
    }
}

int main() {

    int i, j;
    cudaEvent_t start, end;
    float elapsed_time;

    // Allocate memory for the image and edge image on host
    int image = (int) malloc(sizeof(int) * ROWS * COLS);
    int edge = (int) malloc(sizeof(int) * ROWS * COLS);

    // Initialize the image with random values
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            image[i*COLS+j] = rand() % 256;
        }
    }

    // Allocate memory for the image and edge image on device
    int *d_image, *d_edge;
    cudaMalloc((void**) &d_image, sizeof(int) * ROWS * COLS);
    cudaMalloc((void**) &d_edge, sizeof(int) * ROWS * COLS);

    // Copy the image from host to device
    cudaMemcpy(d_image, image, sizeof(int) * ROWS * COLS, cudaMemcpyHostToDevice);

    // Launch the kernel to perform Sobel edge detection
    dim3 dimGrid((COLS-1)/BLOCK_SIZE+1, (ROWS-1)/BLOCK_SIZE+1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    sobel<<<dimGrid, dimBlock>>>(d_image, d_edge, ROWS, COLS);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Copy the edge image from device to host
    cudaMemcpy(edge, d_edge, sizeof(int) * ROWS * COLS, cudaMemcpyDeviceToHost);

    // Write the edge image to a file
FILE *fp;
fp = fopen("edge.pgm", "wb");
fprintf(fp, "P5\n%d %d\n255\n", COLS, ROWS);
for (i = 0; i < ROWS; i++) {
    for (j = 0; j < COLS; j++) {
        fputc((unsigned char)edge[i*COLS+j], fp);
    }
}
fclose(fp);

// Print the execution time
printf("Execution time: %f ms\n", elapsed_time);

// Clean up
free(image);
free(edge);
cudaFree(d_image);
cudaFree(d_edge);

return 0;
}