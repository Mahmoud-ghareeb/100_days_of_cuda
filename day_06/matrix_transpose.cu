#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transpose(float *mat_d, float *out_d, int n, int m)
{
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIdx < n && yIdx < m)
    {
        out_d[yIdx * n + xIdx] = mat_d[xIdx * m + yIdx];
    }
}

int main()
{
    int n = 20;
    int m = 20;
    int size = n*m*sizeof(float);

    float mat[n][m], *out;

    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
            mat[i][j] = j;
        }
    }

    out = (float *)malloc(size);


    float *mat_d, *out_d;

    cudaMalloc(&mat_d, size);
    cudaMalloc(&out_d, size);
    cudaMemcpy(mat_d, mat, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n+blockSize.y-1)/blockSize.y, (n+blockSize.x-1)/blockSize.x);
    transpose<<<gridSize, blockSize>>>(mat_d, out_d, n, m);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    printf("original matrix \n");
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("transposed matrix \n");
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
        {
            printf("%f ", out[i*m+j]);
        }
        printf("\n");
    }

    return 0;
}