#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 1024
#define BLOCK_SIZE 32

_global_ void sobel(int* image, int* edge, int rows, int cols) {
    _shared_ int tile[BLOCK_SIZE+2][BLOCK_SIZE+2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx*blockDim.x + tx;
    int y = by*blockDim.y + ty;

    // Load tile into shared memory
    if (x < cols && y < rows) {
        tile[ty+1][tx+1] = image[y*cols+x];
        if (tx == 0) {
            tile[ty+1][tx] = image[y*cols+(x-1)];
            tile[ty+1][tx+BLOCK_SIZE+1] = image[y*cols+(x+BLOCK_SIZE)];
        }
        if (ty == 0) {
            tile[ty][tx+1] = image[(y-1)*cols+x];
            tile[ty+BLOCK_SIZE+1][tx+1] = image[(y+BLOCK_SIZE)*cols+x];
        }
        if (tx == 0 && ty == 0) {
            tile[ty][tx] = image[(y-1)*cols+(x-1)];
            tile[ty][tx+BLOCK_SIZE+1] = image[(y-1)*cols+(x+BLOCK_SIZE)];
            tile[ty+BLOCK_SIZE+1][tx] = image[(y+BLOCK_SIZE)*cols+(x-1)];
            tile[ty+BLOCK_SIZE+1][tx+BLOCK_SIZE+1] = image[(y+BLOCK_SIZE)*cols+(x+BLOCK_SIZE)];
        }
    }

    __syncthreads();

    // Compute edge values for the tile
    int ex = 0;
    int ey = 0;
    if (x >= 1 && x < cols-1 && y >= 1 && y < rows-1) {
        ex = -1*tile[ty][tx] + tile[ty][tx+2]
             -2*tile[ty+1][tx] + 2*tile[ty+1][tx+2]
             -1*tile[ty+2][tx] + tile[ty+2][tx+2];

        ey = -1*tile[ty][tx] - 2*tile[ty][tx+1] - 1*tile[ty][tx+2]
             + tile[ty+2][tx]  + 2*tile[ty+2][tx+1]  + 1*tile[ty+2][tx+2];

        int val = (int)sqrt(ex*ex + ey*ey);
        if (val > 255) {
            val = 255;
        }
        edge[y*cols+x] = val;
    }

    __syncthreads();
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
