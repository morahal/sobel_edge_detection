#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ROWS 1024
#define COLS 1024

int main(int argc, char **argv) {

    int rank, size;
    int rows_per_process, i, j;
    double start_time, end_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for the image
    int *image = (int*) malloc(sizeof(int) * ROWS * COLS);

    // Initialize the image with random values
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            image[i*COLS+j] = rand() % 256;
        }
    }

    // Calculate the number of rows each process will handle
    rows_per_process = ROWS / size;

    // Allocate memory for the local portion of the image
    int *local_image = (int*) malloc(sizeof(int) * rows_per_process * COLS);

    // Scatter the image to all processes
    MPI_Scatter(image, rows_per_process*COLS, MPI_INT, local_image, rows_per_process*COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Apply Sobel edge detection to the local portion of the image
    int gx, gy, val;
    int *edge = (int*) malloc(sizeof(int) * rows_per_process * COLS);
    for (i = 1; i < rows_per_process-1; i++) {
        for (j = 1; j < COLS-1; j++) {
            gx = -1*local_image[(i-1)*COLS+j-1] + local_image[(i-1)*COLS+j+1]
                 -2*local_image[i*COLS+j-1]     + 2*local_image[i*COLS+j+1]
                 -1*local_image[(i+1)*COLS+j-1] + local_image[(i+1)*COLS+j+1];

            gy = -1*local_image[(i-1)*COLS+j-1] - 2*local_image[(i-1)*COLS+j] - 1*local_image[(i-1)*COLS+j+1]
                 + local_image[(i+1)*COLS+j-1]  + 2*local_image[(i+1)*COLS+j]  + 1*local_image[(i+1)*COLS+j+1];

            val = (int)sqrt(gx*gx + gy*gy);
            if (val > 255) {
                val = 255;
            }
            edge[i*COLS+j] = val;
        }
    }

    // Gather the edges from all processes
    MPI_Gather(edge, rows_per_process*COLS, MPI_INT, image, rows_per_process*COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the execution time
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Execution time: %f seconds\n", end_time-start_time);
    }

    // Clean up
    free(image);
    free(local_image);
    free(edge);

    // Finalize MPI
    MPI_Finalize();

   
    return 0;
}
