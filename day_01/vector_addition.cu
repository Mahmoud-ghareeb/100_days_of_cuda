#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecadd(float *a, float *b, float *c, int n)
{
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }

}

int main()
{
    int n = 6;
    float a_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float b_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float c_h[n];

    float *a_d, *b_d, *c_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

    vecadd<<<ceil(n/1024.0), 1024>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++)
    {
        printf("%f \n", c_h[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}