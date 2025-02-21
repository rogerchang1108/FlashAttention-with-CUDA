#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
// Remove l, m Array

void input(char *input_filename);
void output(char *output_filename);
__global__ void flash_attention_kernel(const float4* q, const float4* k, const float4* v, const int N, const int d,
                    const int tc, const int tr, const int bc, const int br, const float softmax_scale,
                    float *o);

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
    const float softmax_scale = rsqrtf(d);

    const int vec_d = d / 4;
    const int sram_size = (2 * bc * vec_d * sizeof(float4)) + (br * vec_d * sizeof(float4)) + (bc * br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    float4 *h_q = (float4*)malloc(B * N * vec_d * sizeof(float4));
    float4 *h_k = (float4*)malloc(B * N * vec_d * sizeof(float4));
    float4 *h_v = (float4*)malloc(B * N * vec_d * sizeof(float4));

    for (int i = 0; i < B * N * vec_d; i++) {
        int base = i << 2;
        h_q[i] = make_float4(Q[base], Q[base+1], Q[base+2], Q[base+3]);
        h_k[i] = make_float4(K[base], K[base+1], K[base+2], K[base+3]);
        h_v[i] = make_float4(V[base], V[base+1], V[base+2], V[base+3]);
    }

    float4 *d_q, *d_k, *d_v;
    float *d_o;
    
    cudaMalloc(&d_q, B * N * vec_d * sizeof(float4));
    cudaMalloc(&d_k, B * N * vec_d * sizeof(float4));
    cudaMalloc(&d_v, B * N * vec_d * sizeof(float4));
    cudaMalloc(&d_o, B * N * d * sizeof(float));

    cudaMemcpy(d_q, h_q, B * N * vec_d * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, B * N * vec_d * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, B * N * vec_d * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0x00, B * N * d * sizeof(float));

    dim3 grid_dim(B, tr);
    dim3 block_dim(br);
    
    flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_q, d_k, d_v, N, d, tc, tr, bc, br, softmax_scale, d_o
    );

    cudaMemcpy(O, d_o, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_q);
    free(h_k);
    free(h_v);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
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

__global__ void flash_attention_kernel(const float4* q, const float4* k, const float4* v, const int N, const int d,
                    const int tc, const int tr, const int bc, const int br, const float softmax_scale,
                    float *o) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x;

    const int vec_d = d / 4;
    int qkv_offset = (bx * N * vec_d);

    extern __shared__ float4 sram[];
    int tile_size_r = br * vec_d;
    int tile_size_c = bc * vec_d;
    float4* qi = sram;
    float4* kj = &sram[tile_size_r];
    float4* vj = &sram[tile_size_r + tile_size_c];
    float* sij = (float*)&sram[tile_size_r + tile_size_c * 2];

    int i = by;
    // Load Q
    int step = br / vec_d;
    for (int x = 0; x < br; x += step) {
        qi[(x * vec_d) + tx] = q[qkv_offset + (tile_size_r * i) + (x * vec_d) + tx];
    }

    float mi_old = 0.0f;
    float li_old = 0.0f;
    
    for (int j = 0; j < tc; ++j) {
        for (int x = 0; x < bc; x += step) {
            kj[(x * vec_d) + tx] = k[qkv_offset + (tile_size_c * j) + (x * vec_d) + tx];
            vj[(x * vec_d) + tx] = v[qkv_offset + (tile_size_c * j) + (x * vec_d) + tx];
        }
        __syncthreads();

        // QKDotAndScalar(sij) && RowMax(mij)
        float mij = FLT_MIN;
        
        for (int y = 0; y < bc; ++y) {
            float sum = 0.0f;
            for (int x = 0; x < vec_d; ++x) {
                float4 q4 = qi[(tx * vec_d) + x];
                float4 k4 = kj[(y * vec_d) + x];
                sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
            sum *= softmax_scale;
            sij[(tx * bc) + y] = sum;
            mij = fmaxf(mij, sum);
        }
        // MinusMaxAndExp(pij->sij) && RowSum(lij)
        float lij = 0.0f;
        for (int y = 0; y < bc; ++y) {
            sij[(tx * bc) + y] = __expf(sij[(tx * bc) + y] - mij);
            lij += sij[(tx * bc) + y];
        }
        // UpdateMiLiOi
        // Compute new m and l
        float mi_new = fmaxf(mi_old, mij);
        float exp_old = __expf(mi_old - mi_new);
        float exp_new = __expf(mij - mi_new);
        float li_new = (exp_old * li_old) + (exp_new * lij);

        float scale = __fdividef(1.0f, li_new);
        float old_scale = li_old * exp_old * scale;
        float exp_scale = exp_new * scale;

        for (int x = 0; x < vec_d; ++x) {
            float4 pv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int y = 0; y < bc; ++y) {
                float s = sij[(tx * bc) + y];
                float4 v4 = vj[y * vec_d + x];
                pv.x += s * v4.x;
                pv.y += s * v4.y;
                pv.z += s * v4.z;
                pv.w += s * v4.w;
            }
            
            int base_idx = (qkv_offset + (tile_size_r * i) + (tx * vec_d) + x) << 2;
            
            o[base_idx] = old_scale * o[base_idx] + exp_scale * pv.x;
            o[base_idx + 1] = old_scale * o[base_idx + 1] + exp_scale * pv.y;
            o[base_idx + 2] = old_scale * o[base_idx + 2] + exp_scale * pv.z;
            o[base_idx + 3] = old_scale * o[base_idx + 3] + exp_scale * pv.w;
        }

        mi_old = mi_new;
        li_old = li_new;
        __syncthreads();
    }
}