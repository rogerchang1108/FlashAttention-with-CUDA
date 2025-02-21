#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
// Baseline

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);

__global__ void QKDotAndScalarKernel(float *out, float *q, float *k, int br, int bc, int d, float scalar);
__global__ void RowMaxKernel(float *out, float *in, int br, int bc);
__global__ void MinusMaxAndExpKernel(float *out, float *in, float *mx, int br, int bc);
__global__ void RowSumKernel(float *out, float *in, int br, int bc);
__global__ void UpdateMiLiOiKernel(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d);

__device__ float _max(float a, float b) { return a > b ? a : b; }

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    double start, end, input_end, output_start;
    start = getTimeStamp();

    input(argv[1]);

    input_end = getTimeStamp();

    for (int i = 0; i < B; i++) {
        flash_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

    output_start = getTimeStamp();

    output(argv[2]);

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Total Time: %.3f seconds\n", end - start);
    printf("Input Time: %.3f seconds\n", input_end - start);
    printf("Computation Time: %.3f seconds\n", output_start - input_end);
    printf("Output Time: %.3f seconds\n", end - output_start);
    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

void flash_attention(float *q, float *k, float *v, float *o) {
    float *l, *m;
    cudaMallocHost(&l, N * sizeof(float));
    cudaMallocHost(&m, N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int br = 32, bc = 32;
    int tr = N / br, tc = N / bc;
    dim3 blockDim(br, bc);

    float *d_kj, *d_vj, *d_qi, *d_oi, *d_li, *d_mi;
    cudaMalloc(&d_kj, bc * d * sizeof(float));
    cudaMalloc(&d_vj, bc * d * sizeof(float));
    cudaMalloc(&d_qi, br * d * sizeof(float));
    cudaMalloc(&d_oi, br * d * sizeof(float));
    cudaMalloc(&d_li, br * sizeof(float));
    cudaMalloc(&d_mi, br * sizeof(float));

    float *d_sij, *d_mij, *d_pij, *d_lij;
    cudaMalloc(&d_sij, br * bc * sizeof(float));
    cudaMalloc(&d_mij, br * sizeof(float));
    cudaMalloc(&d_pij, br * bc * sizeof(float));
    cudaMalloc(&d_lij, br * sizeof(float));

    for (int j = 0; j < tc; j++) {
        cudaMemcpy(d_kj, k + j * bc * d, bc * d * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vj, v + j * bc * d, bc * d * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 0; i < tr; i++) {
            cudaMemcpy(d_qi, q + i * br * d, br * d * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_oi, o + i * br * d, br * d * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_li, l + i * br, br * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mi, m + i * br, br * sizeof(float), cudaMemcpyHostToDevice);

            QKDotAndScalarKernel<<<1, blockDim>>>(d_sij, d_qi, d_kj, br, bc, d, 1.0 / sqrt(d));
            RowMaxKernel<<<1, br>>>(d_mij, d_sij, br, bc);
            MinusMaxAndExpKernel<<<1, blockDim>>>(d_pij, d_sij, d_mij, br, bc);
            RowSumKernel<<<1, br>>>(d_lij, d_pij, br, bc);
            UpdateMiLiOiKernel<<<1, blockDim>>>(d_mi, d_li, d_oi, d_mij, d_lij, d_pij, d_vj, br, bc, d);

            cudaMemcpy(o + i * br * d, d_oi, br * d * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(l + i * br, d_li, br * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(m + i * br, d_mi, br * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(d_sij);
    cudaFree(d_mij);
    cudaFree(d_pij);
    cudaFree(d_lij);

    cudaFree(d_kj);
    cudaFree(d_vj);
    cudaFree(d_qi);
    cudaFree(d_oi);
    cudaFree(d_li);
    cudaFree(d_mi);

    cudaFree(l);
    cudaFree(m);
}

__global__ void QKDotAndScalarKernel(float *out, float *q, float *k, int br, int bc, int d, float scalar) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    float sum = 0.0f;
    for (int t = 0; t < d; ++t) {
        sum += q[i * d + t] * k[j * d + t];
    }
    out[i * bc + j] = sum * scalar;
}

__global__ void RowMaxKernel(float *out, float *in, int br, int bc) {
    int i = threadIdx.x;

    float max_val = FLT_MIN;
    for (int j = 0; j < bc; j++) {
        max_val = _max(max_val, in[i * bc + j]);
    }
    out[i] = max_val;
}

__global__ void MinusMaxAndExpKernel(float *out, float *in, float *mx, int br, int bc) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    out[i * bc + j] = exp(in[i * bc + j] - mx[i]);
}

__global__ void RowSumKernel(float *out, float *in, int br, int bc) {
    int i = threadIdx.x;

    float sum = 0.0f;
    for (int j = 0; j < bc; j++) {
        sum += in[i * bc + j];
    }
    out[i] = sum;
}

__global__ void UpdateMiLiOiKernel(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    __shared__ float mi_new[32];
    __shared__ float li_new[32];

    if (j == 0) {
        mi_new[i] = _max(mi[i], mij[i]);
        li_new[i] = exp(mi[i] - mi_new[i]) * li[i] + exp(mij[i] - mi_new[i]) * lij[i];
    }
    __syncthreads();

    for (int k = 0; k < d; k += 32) {
        float pv = 0.0f;
        for (int t = 0; t < bc; t++) {
            pv += pij[i * bc + t] * vj[t * d + (k + j)];
        }
        oi[i * d + (k + j)] = (li[i] * exp(mi[i] - mi_new[i]) * oi[i * d + (k + j)] + exp(mij[i] - mi_new[i]) * pv) / li_new[i];
    }

    __syncthreads();
    
    if (j == 0) {
        mi[i] = mi_new[i];
        li[i] = li_new[i];
    }
}