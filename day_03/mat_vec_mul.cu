#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mat_vec_mul(float *a_d, float *b_d, float *c_d, int m)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < m)
    {
        float s = 0.0f;
        for (int i=0; i<m; i++)
        {
            s += a_d[i] * b_d[i*m+c];
        }
        c_d[c] = s;
    }
}

int main()
{
    int n=3, m=3;
    float *a_h, *b_h, *c_h;

    a_h = (float *) malloc(n*sizeof(float));
    b_h = (float *) malloc(n*m*sizeof(float));
    c_h = (float *) malloc(m*sizeof(float));

    for (int i=0; i<n; i++)
    {
        a_h[i] = i*2;
        for (int j=0; j<m; j++)
        {
            b_h[i*m+j] = i*j;
        }
    }

    float *a_d, *b_d, *c_d;

    cudaMalloc((void **) &a_d, n*sizeof(float));
    cudaMalloc((void **) &b_d, n*m*sizeof(float));
    cudaMalloc((void **) &c_d, m*sizeof(float));

    cudaMemcpy(a_d, a_h, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n*m*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16);
    dim3 gridSize(ceil((float)m / 16));

    mat_vec_mul<<<gridSize, blockSize>>>(a_d, b_d, c_d, m);
    
    cudaMemcpy(c_h, c_d, m*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++)
    {
        printf("%f \n", c_h[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}