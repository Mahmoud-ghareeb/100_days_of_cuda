#include <cuda_runtime.h>
#include <stdio.h>

__device__ float E_x(float *shMem, int n, int tid, int idx)
{
    for (int i=blockDim.x/2; i>=1; i /= 2)
    {
        float tmp = 0.0f;
        if (tid+i < blockDim.x)
        {
            tmp = shMem[tid+i];
        }
        __syncthreads();
        shMem[tid] += tmp;
        __syncthreads();
    }

    return shMem[0];
}

__global__ void layer_norm(float *x, float *y, int n)
{
    extern __shared__ float shMem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; 
    float gamma=1.0, beta=1.0, eta=1.0e-8;

    if (idx < n)
    {
        shMem[tid] = x[idx];
        __syncthreads();

        float mean_x = E_x(shMem, n, tid, idx) / n;

        if (idx < n)
            shMem[tid] = x[idx] * x[idx];
        else
            shMem[tid] = 0.0f;
        __syncthreads();

        float mean_xx = E_x(shMem, n, tid, idx) / n;
        float var = mean_xx - (mean_x*mean_x);
        y[idx] = ((x[idx] - mean_x) / sqrt((var + eta))) * gamma + beta;
    }
}

int main()
{
    int n = 16;
    int size = n*sizeof(float);
    float *x, *y;

    x = (float *) malloc(size);
    y = (float *) malloc(size);

    for (int i=0; i<n; i++)
    {
        x[i] = i;
    }

    float *x_d, *y_d;

    cudaMalloc(&x_d, size);
    cudaMalloc(&y_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16);
    dim3 gridSize((n+blockSize.x-1)/blockSize.x);

    layer_norm<<<gridSize, blockSize>>>(x_d, y_d, n);

    cudaMemcpy(y, y_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++)
    {
        printf("%f \n", y[i]);
    }

    cudaFree(x_d);
    cudaFree(y_d);
    free(y);
    free(x);

    return 0;
}