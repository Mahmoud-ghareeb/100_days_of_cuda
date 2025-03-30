#include <cuda_runtime.h>
#include <stdio.h>

__global__ void lora(float *a, float *b, float *c, int m, int n, int r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m*n)
    {
        int row = idx / n;
        int col = idx % n;
        float sum = 0;
        for (int i=0; i<r; i++)
        {
            float temp_a = a[row*r+i];
            float temp_b = b[i*n+col];

            sum += (temp_a * temp_b);
        }

        c[idx] += sum;
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

    int bloackSize = 64;
    int gridSize = (m*n + bloackSize - 1) / bloackSize;

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