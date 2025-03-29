#include <cuda_runtime.h>
#include <stdio.h>

__global__ void safe_softmax(float *a, int m, int n)
{
    extern __shared__ float shMem[];
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y;

    if (y < m)
    {
        float max_v = -INFINITY;

        for (int i=0; i<n; i++)
        {
            shMem[tid*n+i] = a[y*n+i];
        }

        for (int i=0; i<n; i++)
        {
            max_v = fmaxf(shMem[tid*n+i], max_v);
        }
        
        float sum_v = 0.0f;
        for (int i=0; i<n; i++)
        {
            float tmp = expf(shMem[tid*n+i] - max_v);
            shMem[tid*n+i] = tmp;
            sum_v += tmp;
        }

        for (int i=0; i<n; i++)
        {
            shMem[tid*n+i] /= sum_v;
        }

        for (int i=0; i<n; i++)
        {
            a[y*n+i] = shMem[tid*n+i];
        }
    }
}

int main()
{
    int m = 10;
    int n = 5;
    int size = m*n*sizeof(float);
    float *a, *o;

    a = (float *) malloc(size);
    o = (float *) malloc(size);
    
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            a[i * n + j] = i+j;
        }
    }

    float *a_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&a_d, size);

    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    dim3 blockSize(1, 10);
    dim3 gridSize((m+blockSize.y-1)/blockSize.y, (m+blockSize.x-1)/blockSize.x);
    size_t sharedMemSize = sizeof(float) * blockSize.y * n;

    safe_softmax<<<blockSize, gridSize, sharedMemSize>>>(a_d, m, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %.3f ms\n", ms);

    cudaMemcpy(o, a_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            printf("%f  => %f \n", a[i*n+j], o[i*n+j]);
        }
    }

    cudaFree(a_d);
    free(a);
    free(o);

    return 0;
}