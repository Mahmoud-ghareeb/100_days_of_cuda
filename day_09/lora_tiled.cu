#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 1

__global__ void lora(float *a, float *b, float *c, int m, int n, int r)
{
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < m && col <n)
    {
        float sum = 0.0f;
        for (int ph=0; ph<n/TILE_WIDTH; ph++)
        {
            int tiledRow = row;
            int tiledCol = col;
            int tiledK_A = ph * TILE_WIDTH + threadIdx.x;
            int tiledK_B = ph * TILE_WIDTH + threadIdx.y;

            if (tiledRow < m && tiledK_A < r)
                Asub[threadIdx.y][threadIdx.x] = a[tiledRow * r + tiledK_A];
            else
                Asub[threadIdx.y][threadIdx.x] = 0.0f;

            if (tiledK_B < r && tiledCol < n)
                Bsub[threadIdx.y][threadIdx.x] = b[tiledK_B * n + tiledCol];
            else
                Bsub[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k)
            {
                sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
            }

            __syncthreads();
        }

        c[row * n + col] += sum;
    }
}

int main()
{
    float *a, *b, *original_matirx;
    int m = 3;
    int n = 3;
    int r = 2;

    int size_original = m*n*sizeof(float);
    int size_a_lora = m*r*sizeof(float);
    int size_b_lora = n*r*sizeof(float); 

    original_matirx = (float *) malloc(size_original);
    a = (float *) malloc(size_a_lora);
    b = (float *) malloc(size_b_lora);

    for (int i=0; i<m*n; i++)
    {
        original_matirx[i] = 1;
    }

    for (int i=0; i<m*r; i++)
    {
        a[i] = 1;
    }

    for (int i=0; i<n*r; i++)
    {
        b[i] = 1;
    }

    float *a_d, *b_d, *original_matirx_d;

    cudaMalloc(&a_d, size_a_lora);
    cudaMalloc(&b_d, size_b_lora);
    cudaMalloc(&original_matirx_d, size_original);

    cudaMemcpy(a_d, a, size_a_lora, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size_b_lora, cudaMemcpyHostToDevice);
    cudaMemcpy(original_matirx_d, original_matirx, size_original, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 bloackSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((m + bloackSize.y - 1) / bloackSize.y, (n + bloackSize.y - 1) / bloackSize.y);

    lora<<<gridSize, bloackSize>>>(a_d, b_d, original_matirx_d, m, n, r);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %.3f ms\n", ms);

    cudaMemcpy(original_matirx, original_matirx_d, size_original, cudaMemcpyDeviceToHost);

    for (int i=0; i<m*n; i++)
    {
        printf("%f", original_matirx[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(original_matirx_d);
    free(a);
    free(b);
    free(original_matirx);

    return 0;
}