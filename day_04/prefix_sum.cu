#include <cuda_runtime.h>
#include <stdio.h>

__global__ void prefixSum(float *a, float *c, int n)
{
    extern __shared__ float sharedMem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n)
    {
        sharedMem[tid] = a[idx];
        __syncthreads();
    }

    for (int i=1; i<blockDim.x; i*=2)
    {
        float tmp = 0.0f;
        if (tid >= i)
        {
            tmp = sharedMem[tid-i];
        }
        __syncthreads();
        sharedMem[tid] = sharedMem[tid] + tmp;
        __syncthreads();
    }

    c[idx] = sharedMem[tid];
    
}

int main()
{
    int n = 16;
    float *a_h, *c_h;

    a_h = (float *) malloc(n*sizeof(float));
    c_h = (float *) malloc(n*sizeof(float));

    for(int i=0; i<n; i++)
    {
        a_h[i] = i*2;
    }

    float *a_d, *c_d;

    cudaMalloc((void **) &a_d, n*sizeof(float));
    cudaMalloc((void **) &c_d, n*sizeof(float));

    cudaMemcpy(a_d, a_h, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16);
    dim3 gridSize(ceil((float) n / 16));

    prefixSum<<<gridSize, blockSize>>>(a_d, c_d, n);

    cudaMemcpy(c_h, c_d, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i+=1)
    {
        printf("%f \n", c_h[i]);
    }

    cudaFree(a_d);
    cudaFree(c_d);
    free(a_h);
    free(c_h);

    return 0;
}