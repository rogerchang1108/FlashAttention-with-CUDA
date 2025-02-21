#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
// grid_dim(B, tr)

void input(char *input_filename);
void output(char *output_filename);
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v, const int N, const int d,
                    const int tc, const int tr, const int bc, const int br, const float softmax_scale,
                    float *l, float *m, float *o);

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    double start, end, input_end, output_start;
    start = getTimeStamp();

    input(argv[1]);

    input_end = getTimeStamp();

    const int br = 128, bc = (d == 64) ? 16 : 32;
    const int tr = N / br, tc = N / bc;
    const float softmax_scale = 1.0 / sqrt(d);

    const int sram_size = (2 * bc * d * sizeof(float)) + (br * d * sizeof(float)) + (bc * br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    float *d_q, *d_k, *d_v;
    float *d_l, *d_m, *d_o;
    
    cudaMalloc(&d_q, B * N * d * sizeof(float));
    cudaMalloc(&d_k, B * N * d * sizeof(float));
    cudaMalloc(&d_v, B * N * d * sizeof(float));
    cudaMalloc(&d_l, B * N * sizeof(float));
    cudaMalloc(&d_m, B * N * sizeof(float));
    cudaMalloc(&d_o, B * N * d * sizeof(float));

    cudaMemcpy(d_q, Q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_l, 0x00, B * N * sizeof(float));
    cudaMemset(d_m, FLT_MIN, B * N * sizeof(float));
    cudaMemset(d_o, 0x00, B * N * d * sizeof(float));

    dim3 grid_dim(B, tr);
    dim3 block_dim(br);
    
    flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_q, d_k, d_v, N, d, tc, tr, bc, br, softmax_scale, d_l, d_m, d_o
    );

    cudaMemcpy(O, d_o, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(d_o);

    output_start = getTimeStamp();

    output(argv[2]);

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Total Time: %.3f seconds\n", end - start);
    printf("Input Time: %.3f seconds\n", input_end - start);
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

__global__ void flash_attention_kernel(const float* q, const float* k, const float* v, const int N, const int d,
                    const int tc, const int tr, const int bc, const int br, const float softmax_scale,
                    float *l, float *m, float *o) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x;

    int qkv_offset = (bx * N * d);
    int lm_offset = (bx * N);

    extern __shared__ float sram[];
    int tile_size_r = br * d;
    int tile_size_c = bc * d;
    float* qi = sram;
    float* kj = &sram[tile_size_r];
    float* vj = &sram[tile_size_r + tile_size_c];
    float* sij = &sram[tile_size_r + tile_size_c * 2];
    
    for (int j = 0; j < tc; ++j) {
        if(tx < bc){
            for (int x = 0; x < d; ++x) {
                kj[(tx * d) + x] = k[qkv_offset + (tile_size_c * j) + (tx * d) + x];
                vj[(tx * d) + x] = v[qkv_offset + (tile_size_c * j) + (tx * d) + x];
            }
        }
        __syncthreads();

        int i = by;
        // Load Q
        for (int x = 0; x < d; ++x) {
            qi[(tx * d) + x] = q[qkv_offset + (tile_size_r * i) + (tx * d) + x];
        }
        // QKDotAndScalar(sij) && RowMax(mij)
        float mij = FLT_MIN;
        
        for (int y = 0; y < bc; ++y) {
            float sum = 0.0f;
            for (int x = 0; x < d; ++x) {
                sum += qi[(tx * d) + x] * kj[(y * d) + x];
            }
            sum *= softmax_scale;
            sij[(tx * bc) + y] = sum;
            mij = max(mij, sum);
        }
        // MinusMaxAndExp(pij->sij) && RowSum(lij)
        float lij = 0.0f;
        for (int y = 0; y < bc; ++y) {
            sij[(tx * bc) + y] = exp(sij[(tx * bc) + y] - mij);
            lij += sij[(tx * bc) + y];
        }
        // UpdateMiLiOi
        // Compute new m and l
        float mi_old = m[lm_offset + (br * i) + tx];
        float li_old = l[lm_offset + (br * i) + tx];

        float mi_new = max(mi_old, mij);
        float li_new = (exp(mi_old - mi_new) * li_old) + (exp(mij - mi_new) * lij);

        for (int x = 0; x < d; ++x) {
            float pv = 0.0f;
            for (int y = 0; y < bc; ++y) {
                pv += sij[tx * bc + y] * vj[y * d + x];
            }
            
            int base_idx = (qkv_offset + (tile_size_r * i) + (tx * d) + x);

            float scale = 1.0f / li_new;
            float exp_old = exp(mi_old - mi_new);
            float exp_new = exp(mij - mi_new);
            float old_scale = li_old * exp_old * scale;
            float exp_scale = exp_new * scale;
            
            o[base_idx] = old_scale * o[base_idx] + exp_scale * pv;
        }

        m[lm_offset + (br * i) + tx] = mi_new;
        l[lm_offset + (br * i) + tx] = li_new;
        __syncthreads();
    }
}