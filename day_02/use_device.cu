#include <stdio.h>
#include <cuda_runtime.h>

__device__ int add(int a, int b)
{
    return a+b;
}

__global__ void vec_add(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n)
    {
        c[i] = add(a[i], b[i]);
    }

}

int main()
{
    int n = 6;

    int a_h[] = {1, 2, 3, 4, 5, 6};
    int b_h[] = {2, 5, 6, 7, 8, 9};
    int c_h[n];

    int *a_d, *b_d, *c_d;
    int size = n * sizeof(int);

    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

    vec_add<<<ceil(n/1024.0), 1024>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++)
    {
        printf("%d \n", c_h[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}