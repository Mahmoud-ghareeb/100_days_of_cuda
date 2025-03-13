#include <stdio.h>
#include <cuda_runtime.h>

__global__ void oneDconv(float *a, float *b, float *k, int out_width, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < out_width)
    {
        int tmp = 0;
        for (int i=0; i<kernel_size; i++)
        {
            tmp += (a[i+idx] * k[i]);
        }
        b[idx] = tmp;
    }
}

int main()
{
    int kernel_size = 3;
    int width = 10;
    int out_width = width - kernel_size + 1;

    float *mat, *out, *k;

    mat = (float *)malloc(width * sizeof(float));
    out = (float *)malloc(out_width * sizeof(float));
    k = (float *)malloc(kernel_size * sizeof(float));

    for (int i=0; i<kernel_size; i++)
    {
        k[i] = 1;
    }

    for (int i=0; i<width; i++)
    {
        mat[i] = i+1;
    }

    float *mat_d, *out_d, *k_d;

    cudaMalloc(&mat_d, width*sizeof(float));
    cudaMalloc(&out_d, out_width*sizeof(float));
    cudaMalloc(&k_d, kernel_size*sizeof(float));

    cudaMemcpy(mat_d, mat, width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k, kernel_size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16);
    dim3 gridSize((out_width+blockSize.x-1) / blockSize.x);
    oneDconv<<<gridSize, blockSize>>>(mat_d, out_d, k_d, out_width, kernel_size);

    cudaMemcpy(out, out_d, out_width*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<out_width; i++)
    {
        printf("%f ", out[i]);
    }
    
    cudaFree(out_d);
    cudaFree(mat_d);
    cudaFree(k_d);
    free(mat);
    free(out);
    free(k);

    return 0;
}