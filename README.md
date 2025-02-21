# **FlashAttention-with-CUDA Report**

## **1. Title, Name, and Student LinkedIn**
- **Title:** The forward pass of FlashAttention with CUDA.
- **Name:** Roger Chang
- **LinkedIn** [rogerchang1108](https://www.linkedin.com/in/rogerchang1108/)

---

## **2. Implementation**
### **2.1 Overview of the Implementation**

  The code utilizes **CUDA** with **GPUs** to process the forward pass of FlashAttention.

- #### **FlashAttention**:
  <div style="text-align: left;">
    <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/flashattention.png" alt="flashattention" width="450">
  </div>

  In this implementation:
  - There are three integers are the batch size (B), the sequence length (N) and the embedding size (d).

  - There are B batches. Each batch consists of Query (Q), Key (K), and Value (V)  matrices:
    - Query (Q): N * d floating-point numbers
    - Key (K): N * d floating-point numbers
    - Value (V): N * d floating-point numbers
  
  - The ranges for the input are:
    - 2 ≤ B ≤ 14000
    - N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
    - d ∈ {32, 64}
    - -3.0 ≤ Q𝑖,𝑗 , K𝑖,𝑗, V𝑖,𝑗 ≤ 3.0
    - B * N * d < 56000000

- #### **Device Query of the GPU**
    ```
    Device 0: "NVIDIA GeForce GTX 1080"
    CUDA Driver Version / Runtime Version          12.6 / 12.6
    CUDA Capability Major/Minor version number:    6.1
    Total amount of global memory:                 8107 MBytes (8500871168 bytes)
    (20) Multiprocessors, (128) CUDA Cores/MP:     2560 CUDA Cores
    GPU Max Clock rate:                            1734 MHz (1.73 GHz)
    Memory Clock rate:                             5005 Mhz
    Memory Bus Width:                              256-bit
    L2 Cache Size:                                 2097152 bytes
    Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
    Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
    Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
    Total amount of constant memory:               65536 bytes
    Total amount of shared memory per block:       49152 bytes
    Total number of registers available per block: 65536
    Warp size:                                     32
    Maximum number of threads per multiprocessor:  2048
    Maximum number of threads per block:           1024
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
    ```
 
### **2.2 How the FlashAttention forward pass implemented using CUDA**
- #### **Notice**
  
  For the following examples, I will use easier way to demonstrate. There might be some difference in the real implementation of my code. The details will be shown in the *4.2 Optimization* part.

- #### **Matrix Blocking and Tiling**
  - The input matrices Q, K, and V are divided into smaller tiles to maximize shared memory usage.

  - Each **GPU block** processes tiles, each tile with size `br` * `bc` ( `br`: rows, `bc`: columns) to efficiently compute results while staying within shared memory limits.

    - The GPU block size is `br`, which means each thread processes size `bc` of one tile each round.
  
  - `tc` and `tr` represent the number of tiles across the columns and rows.
    ```cpp
    const int tr = N / br, tc = N / bc;
    ```

- #### **Shared Memory (SRAM) Usage**
  - Shared memory is used to store:
     - A tile of Q: Q𝑖 (br * d).
     - A tile of K and V: K𝑗 (bc * d) and V𝑗 (bc * d).
     - Temporary storage for the attention scores S𝑖,𝑗 (bc * br).

  - The total requested shared memory size:
    ```
    const int sram_size = (2 * bc * d * sizeof(float)) + (br * d * sizeof(float)) + (bc * br * sizeof(float))
    ```

  - To maximize thread usage while keeping tiling easy to manage:
    
    - Block Row (`br`): Set to 128 ,which corresponds to the smallest size of `N` in the given input data.

    - Block Column (`bc`): Determined by the given `d`:
      - `d == 64` : Set `bc = 16`
      - `d == 32` : Set `bc = 32`

      These configurations ensure that the requested shared memory size does not exceed the shared memory limit of the GTX 1080 (49152 bytes).

    - Shared Memory Usage:
      - `d == 64` : 49152 bytes
      - `d == 32` : 40960 bytes

- #### **Intermediate Results (𝑚 and ℓ) Computing**
  -  For each K𝑗 and V𝑗:

    - Compute Q𝑖 * K𝑗^T and multiply softmax_scale (1 / sqrt(d)) to get attention score S𝑖,𝑗.
    - And find the row-wise max 𝑚𝑖𝑗 of S𝑖,𝑗.
      ```cpp
      float mij = FLT_MIN;
      for (int y = 0; y < bc; ++y) {
          float sum = 0.0f;
          for (int x = 0; x < d; ++x) {
              sum += qi[(tx * d) + x] * kj[(y * d) + x];
          }
          sum *= softmax_scale;
          sij[(bc * tx) + y] = sum;

          mij = max(sum, mij);
      }
      ```

    - Upate S𝑖,𝑗 by exponential function minus row max 𝑚𝑖𝑗, S𝑖,𝑗 is now the **softmax(Q * K^T)**
    - And compute the row sum of updated S𝑖,𝑗 as ℓ𝑖𝑗
      ```cpp
      float lij = 0.0f;
      for (int y = 0; y < bc; ++y) {
          sij[(bc * tx) + y] = exp(sij[(bc * tx) + y] - mij);

          lij += sij[(bc * tx) + y];
      }
      ```
    
    - Comput the new 𝑚𝑖, ℓ𝑖
    - And update the 𝑚𝑖, ℓ𝑖 after outputing the O𝑖
      ```cpp
      float mi_old = m[lm_offset + (br * i) + tx];
      float li_old = l[lm_offset + (br * i) + tx];

      float mi_new = max(mi_old, mij);
      float li_new = (exp(mi_old - mi_new) * li_old) + (exp(mij - mi_new) * lij);

      // Computing and Outputing the O𝑖
      // ...

      m[lm_offset + (br * i) + tx] = mi_new;
      l[lm_offset + (br * i) + tx] = li_new;
      ```

### **2.3 How Q, K, and V are divided into blocks and processed in parallel**
- #### **Matrix Dimensions**
  - **Matrix Shapes**
    - Q: `B * N * d`

    - K: `B * N * d`

    - V: `B * N * d`

    - `B` is the batch size, `N` is the sequence length, and `d` is the feature dimension.
    
  - **Division into Data Blocks**:
    - `br`: Block row size (number of rows of Q processed by a GPU block).

    - `bc`: Block column size (number of columns of K and V processed per iteration).

- #### **Parallel Processing**
  - **Tile-Based Computation**
    - Q:
       - A tile of size `br * d` is loaded into shared memory for the corresponding row indices.

    - K, V:
       - A tile of K (size `bc * d`) and V (size `bc * d`) is loaded into shared memory for the corresponding column indices.

  - **Block-Level Parallelism**
    - Each **GPU block** processes a tile of Q with `tc` tiles of K, V.

    - GPU blocks are organized into a **Grid**:
      - `bx`: Index along the batch dimension `B`.

      - `by`: Index along the row tiles (`tr = N / br`).

  - **Thread-Level Parallelism**
    - Within each GPU block:
      - `br` threads process rows of Q.

      - Each thread computes interactions between one row of Q and all columns of K, V within the block.

- #### **Crossponding Code (Not Optimized Version)**
  ```cpp
  __global__ void flash_attention_kernel( /*...*/ ) {
      int bx = blockIdx.x; // Which layer(B) is the thread
      int by = blockIdx.y; // Which tile(tr) is the thread
      int tx = threadIdx.x; // Which row of the Q the thread is processing

      int qkv_offset = (bx * N * d); // Offset for Q,K,V to the correct layer

      extern __shared__ float sram[];
      int tile_size_r = br * d; // A tile size for Q
      int tile_size_c = bc * d; // A tile size for K, V
      float* qi = sram;
      float* kj = &sram[tile_size_r];
      float* vj = &sram[tile_size_r + tile_size_c];
      float* sij = &sram[tile_size_r + tile_size_c * 2];

      int i = by; // The tile ID for Q
      // Load Qi to SRAM(Shared Memory)
      for (int x = 0; x < d; ++x) {
          qi[(tx * d) + x] = q[qkv_offset + (tile_size_r * i) + (tx * d) + x];
      }
      
      // Process the K, V tile by tile since they are dependent
      for (int j = 0; j < tc; ++j) {
          // Load Kj, Vj to SRAM(Shared Memory)
          if (tx < bc) {
              for (int x = 0; x < d; x++) {
                  kj[(tx * d) + x] = k[qkv_offset + (tile_size_c * j) + (tx * d) + x];
                  vj[(tx * d) + x] = v[qkv_offset + (tile_size_c * j) + (tx * d) + x];
              }
          }
          __syncthreads(); // Wait all thread loaded
          // Computing
          // ...
          __syncthreads(); // Wait all thread finish computing of this tile
      }
  }
  ```

- #### **Illustration of Grid, Blocks, and Threads**
  <div style="text-align: left;">
    <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/parallel_blocks.png" alt="parallel_blocks" width="800">
  </div>

### **2.4 Block Sizes B_r​ and B_c**
- #### **Fit the limit of Shared Memory**
  - As mentioned in *2.2 How the FlashAttention forward pass implemented using CUDA->Shared Memory (SRAM) Usage*:
  
    To maximize thread usage while keeping tiling easy to manage:
    
      - Block Row (`br`): Set to 128 ,which corresponds to the smallest size of `N` in the given input data.

      - Block Column (`bc`): Determined by the given `d`:
        - `d == 64` : Set `bc = 16`
        - `d == 32` : Set `bc = 32`

    These configurations ensure that the requested shared memory size does not exceed the shared memory limit of the GTX 1080 (49152 bytes).

### **2.5 Configurations for CUDA Kernel Launch**
- #### **Number of Threads Per Block**
  ```cpp
  const int br = 128;

  dim3 block_dim(br);
  ```

- #### **Shared Memory Allocation**
  ```cpp
  const int br = 128, bc = (d == 64) ? 16 : 32;

  const int sram_size = (2 * bc * d * sizeof(float)) + (br * d * sizeof(float)) + (bc * br * sizeof(float));
  ```

- #### **Grid Dimensions**
  ```cpp
  const int br = 128;
  const int tr = N / br;

  dim3 grid_dim(B, tr);
  ```

- #### **The Kernel Function**
  ```cpp
  flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
      d_q, d_k, d_v, N, d, tc, tr, bc, br, softmax_scale, d_l, d_m, d_o
  );
  ```

### **2.6 Justify the Choices and Relationship between Blocking Factors and the SRAM Size**
- #### **Number of Threads Per Block**
  - According to the given information: 

    `N ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}`

    And rows of Q's computation are independent in the FlashAttention.

  - Decide use `br` as block size (number fo threads) and choose the smallest size of `N`: **128** in the given input data for keeping tiling easy to manage (Because `tr = N / br`).

- #### **SRAM (Shared Memory) Size**
  - The limitation of GTX 1080 shared memory is 49152 bytes.
    
    - Shared memory is used to store:
      - A tile of Q: Q𝑖 (br * d).
      - A tile of K and V: K𝑗 (bc * d) and V𝑗 (bc * d).
      - Temporary storage for the attention scores S𝑖,𝑗 (bc * br).

    - So the requested shared memory is: 
        
      `(br * d * sizeof(float)) + (2 * bc * d * sizeof(float)) + (bc * br * sizeof(float))` <= 49152 bytes

      ,which is: `(br * d) + (2 * bc * d) + (bc * br)` <= 12288

    - According to the given information: `d ∈ {32, 64}`

    - When `d == 64`:

      `bc` <= (12288 - 128 * 64) / (2 * 64 + 128)

      ,which is `bc` <= 16.

    - When `d == 32`:

      `bc` <= (12288 - 128 * 32) / (2 * 32 + 128)

      ,which is `bc` <= 42.66.

    - `bc` should be able to divide `N` for keeping tiling easy to manage (Because `tc = N / bc`).
    
    - After experiment, choosing the following combition will achieve the best performance.
      - `d == 64` : Set `bc = 16`
      - `d == 32` : Set `bc = 32`

- #### **Relationship**
  - Justify that:
    - `br` is related to the `N` and **block size** (number fo threads).

    - `bc` is related to the **SRAM** (Shared Memory) size which is calculate by `br`, `bc`, `d`.

---

## **3. Profiling Results**
### **3.1 NVIDIA Profiler**
- #### Occupancy, SM Efficiency, Shared Memory Load/Store Throughput, Global Load/Store Throughput
  - Use the following command to get the `nvprof` metric result. Take `t11` (`(B, N, d): (13600, 128, 32)`) provided by course as input.

    ```
    srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
    ```
  
  - **Result**

    ```
    Invocations                               Metric Name                        Metric Description         Min         Max         Avg
    Device "NVIDIA GeForce GTX 1080 (0)"
        Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
              1                        achieved_occupancy                        Achieved Occupancy    0.124999    0.124999    0.124999
              1                             sm_efficiency                   Multiprocessor Activity      98.45%      98.45%      98.45%
              1                    shared_load_throughput             Shared Memory Load Throughput  1265.0GB/s  1265.0GB/s  1265.0GB/s
              1                   shared_store_throughput            Shared Memory Store Throughput  77.133GB/s  77.133GB/s  77.133GB/s
              1                            gld_throughput                    Global Load Throughput  138.90GB/s  138.90GB/s  138.90GB/s
              1                            gst_throughput                   Global Store Throughput  123.41GB/s  123.41GB/s  123.41GB/s
    ```

  - **Observations:**
    - **Achieved Occupancy**: The achieved occupancy is only 12.49%, which indicates underutilization of the GPU's available threads (128/ 1024).

    - **SM Efficiency**: Multiprocessor activity is very high at 98.45%, showing that the active warps are well-utilized during kernel execution.

    - **Shared Memory Throughput**:
      - Load: 1265.0 GB/s, indicating high utilization for reads.
      - Store: Significantly lower at 77.133 GB/s.

    - **Global Memory Throughput**:
      - Load: 138.90 GB/s and store throughput: 123.41 GB/s, which are higher compared to shared memory store throughput.

---

## **4. Experiment & Analysis**
### **4.1 Methodology**
- #### **System Spec**
  - **Apollo-GPU Server** (Platform provided by the course)
    
    - 8 NVIDIA GPU VMs: each with 2x GTX 1080 GPUs
    
    - 1 AMD GPU VM: with 3x AMD Instinct MI210 GPUs
    
    - Limitations
      - 1 GPU or 2 GPUs per Job
      - 2 CPU cores per GPU (i.e. 1 GPU -> 2 cores, 2 GPUs -> 4 cores)
      - 2 Jobs per User

- #### **Performance Metrics**
  - **NVIDIA Profiler**
    - Utilized `nvprof` with flags to monitor metrics such as utilization and identify potential bottlenecks.

    - Run on the slurm partition nvidia.
      - For Example:
        ```cpp
        // Profiling of time
        srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof ./hw4 <input_file> <output_file>

        // Profiling of some metrics
        srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics <metrics> ./hw4 <input_file> <output_file>
        ```

  - **Time**
    - Use `gettimeofday` to calculate some time such as **Total Time**, **I/O Time**.
      - For Example, **Total Time**:
        ```cpp
        #include <sys/time.h>
        
        double getTimeStamp() {
            struct timeval tv;
            gettimeofday( &tv, NULL );
            return (double) tv.tv_usec/1000000 + tv.tv_sec;
        }

        int main(int argc, char* argv[]){
            double start, end;
            start = getTimeStamp();

            input(argv[1]);
            // Load Data and Do the FlashAttention
            // ...
            output(argv[2]);

            end = getTimeStamp();
            printf("Time: %.3f seconds\n", end - start);
            return 0;
        }
        ```  

  - **Google Sheet**
    - Compute the values and draw the plots.

### **4.2 Optimization**
#### **4.2.1 Kerenl Baseline**
  - Make the sequential version into kernel functirons version. For each function, make it run in kernel:

    ```cpp
    // Sequential
    void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar);
    void RowMax(float *out, float *in, int br, int bc);
    void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
    void RowSum(float *out, float *in, int br, int bc);
    void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc);
    ```

  - into:

    ```cpp
    // Baseline
    __global__ void QKDotAndScalarKernel(float *out, float *q, float *k, int br, int bc, int d, float scalar);
    __global__ void RowMaxKernel(float *out, float *in, int br, int bc);
    __global__ void MinusMaxAndExpKernel(float *out, float *in, float *mx, int br, int bc);
    __global__ void RowSumKernel(float *out, float *in, int br, int bc);
    __global__ void UpdateMiLiOiKernel(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d);
    ```

#### **4.2.2 Kernel Fusing and Tiling + Shared Memory**
  - Make all the kernel functirons into one `flash_attention_kernel` function. 
  
  - As mentioned in *2.3 How Q, K, and V are divided into blocks and processed in parallel*. Tile the Q (`tr` tiles), K (`tc` tiles), V(`tc` tiles) blocks to load them in the shared memory.

    ```cpp
    int main(int argc, char *argv[]) {
        // Read Input
        // Initialize
        const int br = 32, bc = 32;
        const int sram_size = (2 * bc * d * sizeof(float)) + (br * d * sizeof(float)) + (bc * br * sizeof(float));
        dim3 grid_dim(B);
        dim3 block_dim(br);
      
        flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
            d_q, d_k, d_v, N, d, tc, tr, bc, br, softmax_scale, d_l, d_m, d_o
        );

        // Free device meomry 
        /// Write Output
        return 0;
    }
    
    __global__ void flash_attention_kernel(const float* q, const float* k, const float* v, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale, float *l, float *m, float *o) {
        // Initialize
        // ...
        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            for (int i = 0; i < tr; ++i) {
                // Load a tile of Q into shared memory
                // ...
                // QKDotAndScalar(sij) && RowMax(mij)
                // ...
                // MinusMaxAndExp(pij) && RowSum(lij)
                // ...
                // UpdateMiLiOi
                // ...
            }
            __syncthreads();
        }
    }
    ```

#### **4.2.3 Maximum Shared Memory Usage**
  - As mentioned in *2.2 How the FlashAttention forward pass implemented using CUDA->Shared Memory (SRAM) Usage*, to maximum the shared memory usage:

    ```cpp
    const int br = 128, bc = (d == 64) ? 16 : 32;
    
    const int sram_size = (2 * bc * d * sizeof(float)) + (br * d * sizeof(float)) + (bc * br * sizeof(float));
    ```

#### **4.2.4 Split Tiles of Q into Blocks**
  - As mentioned in *2.3 How Q, K, and V are divided into blocks and processed in parallel*, split tiles of Q into blocks.

    ```cpp
    int main(int argc, char *argv[]) {
        // Read Input
        // Initialize
        dim3 grid_dim(B, tr);
        dim3 block_dim(br);
      
        flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
            d_q, d_k, d_v, N, d, tc, tr, bc, br, softmax_scale, d_l, d_m, d_o
        );

        // Free device meomry 
        /// Write Output
        return 0;
    }
    
    __global__ void flash_attention_kernel(const float* q, const float* k, const float* v, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale, float *l, float *m, float *o) {
        int by = blockIdx.y;
        // Initialize
        // ...
        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            int i = by;
            // Load a tile of Q (Qi) into shared memory
            // ...
            // QKDotAndScalar(sij) && RowMax(mij)
            // ...
            // MinusMaxAndExp(pij) && RowSum(lij)
            // ...
            // UpdateMiLiOi
            // ...
            __syncthreads();
        }
    }
    ```

#### **Compare Sequential and 4.2.1 ~ 4.2.4 Versions**
  - **Experimental Method**
    - **Test Case Description**
    
      - **t10** provide by the course
        - (B, N, d): (10, 2048, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t10 test.out
      ```

  - **Profiling Result**

    | Version                  | Total (s) | Input (s) | Computing (s) | Output (s) |
    |--------------------------|-----------|-----------|---------------|------------|
    | Sequential               | 21.651    | 0.025     | 21.61         | 0.016      |
    | Kernel Baseline          | 4.339     | 0.017     | 4.305         | 0.016      |
    | Fusing & Tiling + SM     | 1.016     | 0.017     | 0.984         | 0.015      |
    | Maximum SM               | 0.479     | 0.03      | 0.434         | 0.015      |
    | Split Qi into Blocks     | 0.277     | 0.015     | 0.247         | 0.015      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_1.png" alt="opt_1" width="600">
    </div>

    - `nvprof` result of **Kernel Fusing and Tiling + Shared Memory**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float const *, float const *, float const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.015625    0.015625    0.015625
                1                             sm_efficiency                   Multiprocessor Activity      49.96%      49.96%      49.96%
                1                    shared_load_throughput             Shared Memory Load Throughput  815.03GB/s  815.03GB/s  815.03GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  24.895GB/s  24.895GB/s  24.895GB/s
                1                            gld_throughput                    Global Load Throughput  6.2358GB/s  6.2358GB/s  6.2358GB/s
                1                            gst_throughput                   Global Store Throughput  3.0760GB/s  3.0760GB/s  3.0760GB/s
      ```

    - `nvprof` result of **Maximum Shared Memory Usage**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float const *, float const *, float const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.062500    0.062500    0.062500
                1                             sm_efficiency                   Multiprocessor Activity      49.88%      49.88%      49.88%
                1                    shared_load_throughput             Shared Memory Load Throughput  1856.5GB/s  1856.5GB/s  1856.5GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  93.519GB/s  93.519GB/s  93.519GB/s
                1                            gld_throughput                    Global Load Throughput  37.307GB/s  37.307GB/s  37.307GB/s
                1                            gst_throughput                   Global Store Throughput  18.545GB/s  18.545GB/s  18.545GB/s
      ```

    - `nvprof` result of **Split Tiles of Q into Blocks**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float const *, float const *, float const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.125000    0.125000    0.125000
                1                             sm_efficiency                   Multiprocessor Activity      99.70%      99.70%      99.70%
                1                    shared_load_throughput             Shared Memory Load Throughput  4110.2GB/s  4110.2GB/s  4110.2GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  245.39GB/s  245.39GB/s  245.39GB/s
                1                            gld_throughput                    Global Load Throughput  92.180GB/s  92.180GB/s  92.180GB/s
                1                            gst_throughput                   Global Store Throughput  41.058GB/s  41.058GB/s  41.058GB/s
      ```

  - **Observation and Analysis**

    - **Performance Improvements**:
      - Significant reduction in compute time across versions: from 21.61 seconds (**Sequential**) to 0.247 seconds (**Split Qi into Blocks**), achieving an ~87x speedup overall.
      - Input and output times remain minimal.
    
    - **Maximum Shared Memory Usage**:
      - In the version, all **Memory Throughput** improve, indicating  much better of using Shared Memory.

    - **Split Tiles of Q into Blocks**:
      - In the version, **SM Efficiency** dramatic improvement, reaching 99.7%, showing near-maximal utilization of the GPU multiprocessors. Also make all **Memory Throughput** improve by the efficient utilizing of SM.

      - **Achieved Occupancy** stays at 12.5% because the implentmation use only 128 threads (Maximum number of threads per block: 1024).

#### **4.2.5 Use Vector Data Type `float4`**
  - Use `float4` to optimize data access and computation. The `float4` type is used for `Q`, `K`, `V`, allowing the kernel to load, store, and compute four values at once. This reduces the number of memory accesses and instructions required for computations.

    ```cpp
    __global__ void flash_attention_kernel(const float4* q, const float4* k, const float4* v, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale, float *l, float *m, float *o) {
        // Initialize
        // ...
        const int vec_d = d / 4;

        extern __shared__ float4 sram[];
        int tile_size_r = br * vec_d;
        int tile_size_c = bc * vec_d;
        float4* qi = sram;
        float4* kj = &sram[tile_size_r];
        float4* vj = &sram[tile_size_r + tile_size_c];
        float* sij = (float*)&sram[tile_size_r + tile_size_c * 2];
        
        for (int j = 0; j < tc; ++j) {
            if (tx < bc) {
                for (int x = 0; x < vec_d; ++x) {
                    kj[(tx * vec_d) + x] = k[qkv_offset + (tile_size_c * j) + (tx * vec_d) + x];
                    vj[(tx * vec_d) + x] = v[qkv_offset + (tile_size_c * j) + (tx * vec_d) + x];
                }
            }
            __syncthreads();

            int i = by;
            // Load a tile of Q (Qi) into shared memory
            for (int x = 0; x < vec_d; ++x) {
                qi[(tx * vec_d) + x] = q[qkv_offset + (tile_size_r * i) + (tx * vec_d) + x];
            }
            // QKDotAndScalar(sij) && RowMax(mij)
            float mij = FLT_MIN;
        
            for (int y = 0; y < bc; ++y) {
                float sum = 0.0f;
                for (int x = 0; x < vec_d; ++x) {
                    float4 q4 = qi[(tx * vec_d) + x]; // Load 4 data in Qi
                    float4 k4 = kj[(y * vec_d) + x];  // Load 4 data in Kj
                    sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w; // Compute 4 data
                }
                sum *= softmax_scale;
                sij[(tx * bc) + y] = sum;
                mij = max(mij, sum);
            }
            // MinusMaxAndExp(pij) && RowSum(lij)
            // ...
            // UpdateMiLiOi
            // // Compute new m and l
            // // ...
            for (int x = 0; x < vec_d; ++x) {
                float4 pv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int y = 0; y < bc; ++y) {
                    float s = sij[(tx * bc) + y];
                    float4 v4 = vj[y * vec_d + x]; // Load 4 data in Vj
                    // Compute 4 data
                    pv.x += s * v4.x;
                    pv.y += s * v4.y;
                    pv.z += s * v4.z;
                    pv.w += s * v4.w;
                }
                
                int base_idx = (qkv_offset + (tile_size_r * i) + (tx * vec_d) + x) << 2;

                float scale = 1.0f / li_new;
                float exp_old = exp(mi_old - mi_new);
                float exp_new = exp(mij - mi_new);
                float old_scale = li_old * exp_old * scale;
                float exp_scale = exp_new * scale;
                
                // Compute 4 data and store to Oi
                o[base_idx] = old_scale * o[base_idx] + exp_scale * pv.x;
                o[base_idx + 1] = old_scale * o[base_idx + 1] + exp_scale * pv.y;
                o[base_idx + 2] = old_scale * o[base_idx + 2] + exp_scale * pv.z;
                o[base_idx + 3] = old_scale * o[base_idx + 3] + exp_scale * pv.w;
            }

            // Update m, l
            //...
            __syncthreads();
        }
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version              | Total (s) | Input (s) | Computing (s) | Output (s) |
    |----------------------|-----------|-----------|---------------|------------|
    | Split Qi with t11    | 1.731     | 0.543     | 0.832         | 0.356      |
    | float4 with t11      | 1.504     | 0.361     | 0.768         | 0.375      |
    | Split Qi with t22    | 7.511     | 0.512     | 6.552         | 0.447      |
    | float4 with t22      | 3.771     | 0.814     | 2.516         | 0.441      |
    | Split Qi with t29    | 8.173     | 0.042     | 8.055         | 0.076      |
    | float4 with t29      | 2.687     | 0.082     | 2.526         | 0.079      |
    | Split Qi with t30    | 6.333     | 0.028     | 6.271         | 0.034      |
    | float4 with t30      | 2.186     | 0.028     | 2.124         | 0.034      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_2.png" alt="opt_2" width="600">
    </div>

    - `nvprof` result of **4.2.5 Use Vector Data Type `float4`**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.124998    0.124998    0.124998
                1                    shared_load_throughput             Shared Memory Load Throughput  3942.2GB/s  3942.2GB/s  3942.2GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  520.15GB/s  520.15GB/s  520.15GB/s
                1                            gld_throughput                    Global Load Throughput  75.713GB/s  75.713GB/s  75.713GB/s
                1                            gst_throughput                   Global Store Throughput  55.181GB/s  55.181GB/s  55.181GB/s
      ==605534== Warning: One or more events or metrics overflowed. Rerun with "--print-gpu-trace" for detail.
      ```

  - **Observation and Analysis**

    - **Performance Improvements**:
      - Using the `float4` vector data type consistently outperforms the previous version across all test cases, especially for larger inputs like `t29` and `t30`.

      - For `t29`, the compute time decreases from 8.055s to 2.526s (~3.2x speedup). Similarly, for `t30`, compute time drops from 6.271s to 2.124s (~2.95x speedup).

    - **Shared Memory Throughput**:
      - **Store throughput** jumps to 520.15 GB/s, showing reduced memory transaction overhead.

#### **4.2.6 Restructure the Code**
  - The `flash_attention_kernel` was restructured to move repetitive operations and calculations outside loops wherever possible. This approach minimizes redundant computations and improves overall performance by reducing the number of instructions executed per thread.

    ```cpp
    __global__ void flash_attention_kernel(const float4* q, const float4* k, const float4* v, const int N, const int d, const int tc, const int tr, const int bc, const int br, const float softmax_scale, float *l, float *m, float *o) {
        // Initialize
        // ...
        // Move the Loading Q out of loop
        for (int x = 0; x < vec_d; ++x) {
            qi[(tx * vec_d) + x] = q[qkv_offset + (tile_size_r * i) + (tx * vec_d) + x];
        }
        
        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            // QKDotAndScalar(sij) && RowMax(mij)
            // ...
            // MinusMaxAndExp(pij->sij) && RowSum(lij)
            // ...
            // UpdateMiLiOi
            // Compute new m and l
            float mi_old = m[lm_offset + (br * i) + tx];
            float li_old = l[lm_offset + (br * i) + tx];

            float mi_new = fmaxf(mi_old, mij);
            // Move exp_old and exp_new out of loop
            float exp_old = __expf(mi_old - mi_new);
            float exp_new = __expf(mij - mi_new);
            float li_new = (exp_old * li_old) + (exp_new * lij);

            // Move scale, old_scale and exp_scale out of loop
            float scale = __fdividef(1.0f, li_new);
            float old_scale = li_old * exp_old * scale;
            float exp_scale = exp_new * scale;

            for (int x = 0; x < vec_d; ++x) {
                // Compute PV
                // ...
                
                int base_idx = (qkv_offset + (tile_size_r * i) + (tx * vec_d) + x) << 2;
                
                // Compute data and store to Oi
                // ...
            }

            m[lm_offset + (br * i) + tx] = mi_new;
            l[lm_offset + (br * i) + tx] = li_new;
            __syncthreads();
        }
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version                | Total (s) | Input (s) | Computing (s) | Output (s) |
    |------------------------|-----------|-----------|---------------|------------|
    | float4 with t11        | 1.504     | 0.361     | 0.768         | 0.375      |
    | Restructure with t11   | 1.477     | 0.36      | 0.74          | 0.377      |
    | float4 with t22        | 3.771     | 0.814     | 2.516         | 0.441      |
    | Restructure with t22   | 3.124     | 0.347     | 2.348         | 0.429      |
    | float4 with t29        | 2.687     | 0.082     | 2.526         | 0.079      |
    | Restructure with t29   | 2.55      | 0.042     | 2.432         | 0.076      |
    | float4 with t30        | 2.186     | 0.028     | 2.124         | 0.034      |
    | Restructure with t30   | 1.927     | 0.046     | 1.846         | 0.035      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_3.png" alt="opt_3" width="600">
    </div>

    - `nvprof` result of **4.2.6 Restructure the Code**

      ```
      ==605878== Profiling application: ./hw4 testcases/t29 test.out
      ==605878== Profiling result:
      ==605878== Metric result:
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.125000    0.125000    0.125000
                1                    shared_load_throughput             Shared Memory Load Throughput  3995.3GB/s  3995.3GB/s  3995.3GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  471.72GB/s  471.72GB/s  471.72GB/s
                1                            gld_throughput                    Global Load Throughput  62.873GB/s  62.873GB/s  62.873GB/s
                1                            gst_throughput                   Global Store Throughput  55.923GB/s  55.923GB/s  55.923GB/s
      ==605878== Warning: One or more events or metrics overflowed. Rerun with "--print-gpu-trace" for detail.
      ```

  - **Observation and Analysis**
    - **Performance Improvements**:
      - The Restructured version improves performance compared to the previous version across all test cases.

        - For t11, compute time decreases from 0.768s to 0.74s (~3.6% improvement).
        - For t22, compute time decreases from 2.516s to 2.348s (~6.7% improvement).
        - For larger cases like t30, compute time drops from 2.124s to 1.846s (~13.1% improvement).

      - The total execution time also reflects these improvements, especially for larger test cases (t29 and t30).
    
    - **Memory Throughput**
      - **Shared Memory Store** and **Global Load** slightly decrease, which might be due to the reduction of repetitive loading of `Q`.

#### **4.2.7 Memory Coalescing**
  - The access to `Q`, `K` and `V` is restructured to ensure that the threads read contiguous chunks of these arrays in a coalesced manner and achieve memory coalescing.

    ```cpp
    int step = br / vec_d;
    // Load Q with Memory Coalescing
    for (int x = 0; x < br; x += step) {
        qi[(x * vec_d) + tx] = q[qkv_offset + (tile_size_r * i) + (x * vec_d) + tx];
    }
    
    for (int j = 0; j < tc; ++j) {
        // Load K, J with Memory Coalescing
        for (int x = 0; x < bc; x += step) {
            kj[(x * vec_d) + tx] = k[qkv_offset + (tile_size_c * j) + (x * vec_d) + tx];
            vj[(x * vec_d) + tx] = v[qkv_offset + (tile_size_c * j) + (x * vec_d) + tx];
        }
        __syncthreads();
        // Computing
        // ...
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version                         | Total (s) | Input (s) | Computing (s) | Output (s) |
    |---------------------------------|-----------|-----------|---------------|------------|
    | Restructure with t11            | 1.477     | 0.36      | 0.74          | 0.377      |
    | Memory Coalescing with t11      | 1.442     | 0.363     | 0.739         | 0.34       |
    | Restructure with t22            | 3.124     | 0.347     | 2.348         | 0.429      |
    | Memory Coalescing with t22      | 3.109     | 0.351     | 2.335         | 0.423      |
    | Restructure with t29            | 2.55      | 0.042     | 2.432         | 0.076      |
    | Memory Coalescing with t29      | 2.514     | 0.025     | 2.412         | 0.077      |
    | Restructure with t30            | 1.927     | 0.046     | 1.846         | 0.035      |
    | Memory Coalescing with t30      | 1.873     | 0.027     | 1.812         | 0.034      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_4.png" alt="opt_4" width="600">
    </div>

    - `nvprof` result of **4.2.7 Memory Coalescing**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*, float*, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.125000    0.125000    0.125000
                1                    shared_load_throughput             Shared Memory Load Throughput  4041.2GB/s  4041.2GB/s  4041.2GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  452.54GB/s  452.54GB/s  452.54GB/s
                1                            gld_throughput                    Global Load Throughput  60.082GB/s  60.082GB/s  60.082GB/s
                1                            gst_throughput                   Global Store Throughput  56.567GB/s  56.567GB/s  56.567GB/s
      ==606736== Warning: One or more events or metrics overflowed. Rerun with "--print-gpu-trace" for detail.
      ```

  - **Observation and Analysis**
    - **Performance Improvements**:
      - The memory coalescing results in a small reduction in Total Time across all test cases compared to the previous version.

    - **Memory Throughput**:
      - **Shared Memory Load Throughput** shows a slight increase, reaching 4041.2 GB/s, indicating better alignment of memory accesses.

      - **Shared Memory Store Throughput** and **Global Load Throughput** see marginal reductions, which may be a trade-off for achieving coalesced memory access patterns and reduced contention.

      - **Global Store Throughput** remains stable, showing no significant performance penalty due to the memory coalescing optimizatio

#### **4.2.8 Remove l, m Array**
  - The `l` and `m` arrays are removed and replaced with registers (`mi_old` and `li_old`). Since each thread operates independently and only needs a single value of `l` and `m` at a time, using registers for these values improves efficiency and reduces memory overhead.

    ```cpp
    __global__ void flash_attention_kernel(/* ... */, const float softmax_scale, float *o) { // Remove l, m
        // Initialize
        // ...
        int i = by;
        // Load a tile of Q (Qi) into shared memory
        // ...
        // Use register to record mi_old, li_old for each thread
        float mi_old = 0.0f;
        float li_old = 0.0f;

        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            // QKDotAndScalar(sij) && RowMax(mij)
            // ...
            // MinusMaxAndExp(pij) && RowSum(lij)
            // ...
            // UpdateMiLi
            float mi_new = fmaxf(mi_old, mij);
            float exp_old = __expf(mi_old - mi_new);
            float exp_new = __expf(mij - mi_new);
            float li_new = (exp_old * li_old) + (exp_new * lij);
            float scale = __fdividef(1.0f, li_new);
            float old_scale = li_old * exp_old * scale;
            float exp_scale = exp_new * scale;

            // UpdateOi
            // ...

            // Update mi, li for next tile of K, J
            mi_old = mi_new;
            li_old = li_new;
            __syncthreads();
        }
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

    - Use the following command to check bank conflicts count.
      ``` 
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof  --events shared_ld_bank_conflict,shared_st_bank_conflict ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version                         | Total (s) | Input (s) | Computing (s) | Output (s) |
    |----------------------------------|-----------|-----------|---------------|------------|
    | Memory Coalescing with t11       | 1.442     | 0.363     | 0.739         | 0.34       |
    | Remove l, m with t11            | 1.455     | 0.34      | 0.741         | 0.374      |
    | Memory Coalescing with t22       | 3.109     | 0.351     | 2.335         | 0.423      |
    | Remove l, m with t22            | 3.193     | 0.381     | 2.373         | 0.439      |
    | Memory Coalescing with t29       | 2.514     | 0.025     | 2.412         | 0.077      |
    | Remove l, m with t29            | 2.502     | 0.025     | 2.441         | 0.036      |
    | Memory Coalescing with t30       | 1.873     | 0.027     | 1.812         | 0.034      |
    | Remove l, m with t30            | 1.865     | 0.025     | 1.806         | 0.034      |


    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_5.png" alt="opt_5" width="600">
    </div>

    - `nvprof` result of **4.2.8 Remove l, m Array**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.125000    0.125000    0.125000
                1                    shared_load_throughput             Shared Memory Load Throughput  4006.6GB/s  4006.6GB/s  4006.6GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  448.66GB/s  448.66GB/s  448.66GB/s
                1                            gld_throughput                    Global Load Throughput  59.131GB/s  59.131GB/s  59.131GB/s
                1                            gst_throughput                   Global Store Throughput  55.647GB/s  55.647GB/s  55.647GB/s
      ==610506== Warning: One or more events or metrics overflowed. Rerun with "--print-gpu-trace" for detail.
      ```

      ```
      Invocations                                Event Name         Min         Max         Avg       Total
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
                1                   shared_ld_bank_conflict  6.7512e+10  6.7512e+10  6.7512e+10  6.7512e+10
                1                   shared_st_bank_conflict  8321499136  8321499136  8321499136  8321499136
      ```

  - **Observation and Analysis**
    - **Performance Improvements**:
      - The **Remove l, m Array** results in a small or even no reduction in Total Time across all test cases compared to the previous version.

    - **Memory Throughput**:
      - **Global Load** and **Global Load** **Global Store** slightly decrease, which might be due to there is no need to load and store `l` and `m` from global memory anymore.

#### **4.2.9 Avoid Bank Conflicts**
  - Change the accessing of `qi` to Avoid Bank Conflicts. By changing the indexing of shared memory in a way, avoids multiple threads in the same warp accessing the same bank.

  - Modify the indexing of `sij` to Avoid Bank Conflicts. The indexing of `sij` ensures that different threads access different memory banks.

  - However, it also needs to modify the way `qi` is stored and `q` is loaded, which prevents memory coalescing on `q`.

    ```cpp
    __global__ void flash_attention_kernel(/* ... */) {
        // Initialize
        // ...
        int i = by;
        for (int x = 0; x < vec_d; ++x) {
            qi[(x * br) + tx] = q[qkv_offset + (tile_size_r * i) + (tx * vec_d) + x]; // Avoid Bank Conflicts
        }

        int step = br / vec_d;
        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            // QKDotAndScalar(sij) && RowMax(mij)
            float mij = FLT_MIN;
        
            for (int y = 0; y < bc; ++y) {
                float sum = 0.0f;
                for (int x = 0; x < vec_d; ++x) {
                    float4 q4 = qi[(x * br) + tx]; // Avoid Bank Conflicts
                    float4 k4 = kj[(y * vec_d) + x];
                    sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
                sum *= softmax_scale;
                sij[(y * br) + tx] = sum; // Avoid Bank Conflicts
                mij = fmaxf(mij, sum);
            }
            // MinusMaxAndExp(pij) && RowSum(lij)
            float lij = 0.0f;
            for (int y = 0; y < bc; ++y) {
                sij[(y * br) + tx] = __expf(sij[(y * br) + tx] - mij); // Avoid Bank Conflicts
                lij += sij[(y * br) + tx]; // Avoid Bank Conflicts
            }
            // UpdateMiLi
            // Compute required variables
            // ...
            for (int x = 0; x < vec_d; ++x) {
                float4 pv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int y = 0; y < bc; ++y) {
                    float s = sij[(y * br) + tx]; // Avoid Bank Conflicts
                    float4 v4 = vj[y * vec_d + x];
                    pv.x += s * v4.x;
                    pv.y += s * v4.y;
                    pv.z += s * v4.z;
                    pv.w += s * v4.w;
                }
                
                // UpdateOi
                // ...
            }
            // Update mi, li for next tile of K, J
            //...
            __syncthreads();
        }
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version                          | Total (s) | Input (s) | Computing (s) | Output (s) |
    |-----------------------------------|-----------|-----------|---------------|------------|
    | Remove l, m with t11              | 1.455     | 0.34      | 0.741         | 0.374      |
    | Avoid Bank Conflicts with t11     | 1.435     | 0.372     | 0.695         | 0.368      |
    | Remove l, m with t22              | 3.193     | 0.381     | 2.373         | 0.439      |
    | Avoid Bank Conflicts with t22     | 2.425     | 0.384     | 1.598         | 0.443      |
    | Remove l, m with t29              | 2.502     | 0.025     | 2.441         | 0.036      |
    | Avoid Bank Conflicts with t29     | 0.844     | 0.027     | 0.737         | 0.08       |
    | Remove l, m with t30              | 1.865     | 0.025     | 1.806         | 0.034      |
    | Avoid Bank Conflicts with t30     | 1.108     | 0.028     | 1.043         | 0.037      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_6.png" alt="opt_6" width="450">
    </div>

    - `nvprof` result of **4.2.9 Avoid Bank Conflicts**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.124998    0.124998    0.124998
                1                             sm_efficiency                   Multiprocessor Activity      98.33%      98.33%      98.33%
                1                    shared_load_throughput             Shared Memory Load Throughput  1923.2GB/s  1923.2GB/s  1923.2GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  65.888GB/s  65.888GB/s  65.888GB/s
                1                            gld_throughput                    Global Load Throughput  223.98GB/s  223.98GB/s  223.98GB/s
                1                            gst_throughput                   Global Store Throughput  210.76GB/s  210.76GB/s  210.76GB/s
      ```

      ```
      Invocations                                Event Name         Min         Max         Avg       Total
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
                1                   shared_ld_bank_conflict           0           0           0           0
                1                   shared_st_bank_conflict           0           0           0           0
      ```

  - **Observation and Analysis**
    - **Performance Improvements**:
      - **Avoid Bank Conflicts** significantly improves performance, especially for larger test cases like `t22`, `t29`, and `t30`. 

      - **Total Time**: The **Total Time** for test case `t29` drops from 2.502s (previous version) to 0.844s (Avoid Bank Conflicts), and for `t30`, it decreases from 1.865s to 1.108s. This demonstrates a substantial reduction in the overall execution time due to better memory access patterns.

      - **Computing Time**: For `t29`, the **Computing Time** decreases from 2.441s to 0.737s, and for `t22`, it drops from 2.373s to 1.598s. This shows that optimizing memory access to avoid bank conflicts leads to more efficient computation, likely due to better use of shared memory and reduced contention.

    - **Memory Throughput**:
      - **Shared Memory Load Throughput** drops significantly from around 4006.6GB/s (in previous versions) to 1923.2GB/s when avoiding bank conflicts. This indicates a more balanced memory access pattern, though it may be accessing fewer memory banks efficiently, suggesting that avoiding conflicts could lead to reduced peak throughput.

      - **Shared Memory Store Throughput** also shows a reduction, dropping to 65.888GB/s, which is consistent with avoiding bank conflicts and optimizing memory access to ensure better alignment for different threads.
      
      - **Global Load Throughput** and **Global Store Throughput** are considerably improved, with values of 223.98GB/s and 210.76GB/s, respectively, which indicates more efficient access to global memory due to optimized memory access patterns.

#### **4.2.10 Use Register on Qi to Extend bc**
  - Use Register to store `qi` for each thread. Since the `d` is either 32 or 64, it will correspond to 8 or 16 `float4` data elements, so declare `float4 qi[16];`.

  - Because `qi` no longer needs to be stored in shared memory, the formula for `sram_size` becomes: `(2 * bc * vec_d * sizeof(float4)) + (bc * br * sizeof(float));`. This size must not exceed 49152 bytes.

    - Keep `br = 128` and adjust `bc` based on the value of `d`:

      - `d == 64`, `bc = 32`
      - `d == 32`, `bc = 64`

    ```cpp
    const int br = 128, bc = (d == 64) ? 32 : 64; // Change the bc size

    const int vec_d = d / 4;
    const int sram_size = (2 * bc * vec_d * sizeof(float4)) + (bc * br * sizeof(float)); // Change the sram size formula
    ```

    ```cpp
    __global__ void flash_attention_kernel(/* ... */) {
        // Initialize
        // ...
        extern __shared__ float4 sram[];
        int tile_size_r = br * vec_d;
        int tile_size_c = bc * vec_d;
        float4* kj = sram;
        float4* vj = &sram[tile_size_c];
        float* sij = (float*)&sram[tile_size_c * 2];
        
        float4 qi[16]; // Store qi in registers instead of shared memory

        int i = by;
        // Load Q into registers
        for (int x = 0; x < vec_d; ++x) {
            qi[x] = q[qkv_offset + (tile_size_r * i) + (tx * vec_d) + x];
        }

        for (int j = 0; j < tc; ++j) {
            // Load a tile of K, V into shared memory
            // ...
            // QKDotAndScalar(sij) && RowMax(mij)
            float mij = FLT_MIN;
        
            for (int y = 0; y < bc; ++y) {
                float sum = 0.0f;
                for (int x = 0; x < vec_d; ++x) {
                    float4 q4 = qi[x]; // Load from registers
                    float4 k4 = kj[(y * vec_d) + x];
                    sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
                sum *= softmax_scale;
                sij[(y * br) + tx] = sum;
                mij = fmaxf(mij, sum);
            }
            // MinusMaxAndExp(pij) && RowSum(lij)
            // ...
            // UpdateMiLiOi
            // ...
            // Update mi, li for next tile of K, J
            //...
            __syncthreads();
        }
    }
    ```

  - **Experimental Method**
    - **Test Case Description**

      - **t11** provide by the course
        - (B, N, d): (13600, 128, 32)

      - **t22** provide by the course
        - (B, N, d): (500, 2048, 64)

      - **t29** provide by the course
        - (B, N, d): (4, 32768, 32)

      - **t30** provide by the course
        - (B, N, d): (2, 32768, 64)
 
  - **Performance Measurement**
    - **Use `gettimeofday` to calculate times**

      - **Total Time**: The time from the beginning of `main()` function to the end of it, right before `return 0`.

      - **Input Time**: The `input()` function time.
          
      - **Computing Time**: The process data part time, including kernel function, `cudaMalloc`, `cudaMemcpy` and so on.

      - **Output Time**: The `output()` function time.
    
    - Use the following command to get the `achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput` result.
      ```
      srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./hw4 testcases/t29 test.out
      ```

  - **Profiling Result**

    | Version                          | Total (s) | Input (s) | Computing (s) | Output (s) |
    |-----------------------------------|-----------|-----------|---------------|------------|
    | Avoid Bank Conflicts with t11     | 1.435     | 0.372     | 0.695         | 0.368      |
    | Qi Register with t11              | 1.4       | 0.353     | 0.658         | 0.389      |
    | Avoid Bank Conflicts with t22     | 2.425     | 0.384     | 1.598         | 0.443      |
    | Qi Register with t22              | 2.224     | 0.349     | 1.435         | 0.44       |
    | Avoid Bank Conflicts with t29     | 0.844     | 0.027     | 0.737         | 0.08       |
    | Qi Register with t29              | 0.8       | 0.057     | 0.668         | 0.075      |
    | Avoid Bank Conflicts with t30     | 1.108     | 0.028     | 1.043         | 0.037      |
    | Qi Register with t30              | 0.969     | 0.058     | 0.878         | 0.033      |

    <div style="text-align: left;">
      <img src="https://raw.githubusercontent.com/rogerchang1108/FlashAttention-with-CUDA/main/img/opt_7.png" alt="opt_7" width="600">
    </div>

    - `nvprof` result of **4.2.10 Use Register on Qi to Extend bc**

      ```
      Invocations                               Metric Name                        Metric Description         Min         Max         Avg
      Device "NVIDIA GeForce GTX 1080 (0)"
          Kernel: flash_attention_kernel(float4 const *, float4 const *, float4 const *, int, int, int, int, int, int, float, float*)
                1                        achieved_occupancy                        Achieved Occupancy    0.124999    0.124999    0.124999
                1                             sm_efficiency                   Multiprocessor Activity      98.45%      98.45%      98.45%
                1                    shared_load_throughput             Shared Memory Load Throughput  1265.0GB/s  1265.0GB/s  1265.0GB/s
                1                   shared_store_throughput            Shared Memory Store Throughput  77.133GB/s  77.133GB/s  77.133GB/s
                1                            gld_throughput                    Global Load Throughput  138.90GB/s  138.90GB/s  138.90GB/s
                1                            gst_throughput                   Global Store Throughput  123.41GB/s  123.41GB/s  123.41GB/s
      ```

  - **Observation and Analysis**
    - **Performance Improvements**:
      - **Total Time**: For larger test cases such as `t29` and `t30`, the total execution time decreases significantly when using registers for `qi`. For instance, with `t29`, the total time drops from 0.844s (previous version) to 0.8s (using registers), and for `t30`, the time decreases from 1.108s to 0.969s.

      - **Computing Time**: The **Computing Time** also improves with the use of registers, as shown by the reduction in time for `t29` from 0.737s to 0.668s and for `t30` from 1.043s to 0.878s. The time reduction is likely due to biger `bc` size, which makes each thread iterate less.

    - **Memory Throughput**:
      - **Shared Memory Throughput**: The **Shared Memory Load Throughput** drops from around 1923.2GB/s (Avoid Bank Conflicts) to 1265GB/s, while the **Shared Memory Store Throughput** increases from 65.888GB/s to 77.133GB/s. This is expected, as data is now stored in registers instead of shared memory, reducing shared memory load throughput but increasing store throughput as the data is written back to global memory from the registers.

      - **Global Throughput**: **Global Load Throughput** and **Global Store Throughput** show moderate improvements, with the load throughput increasing from 223.98GB/s to 138.90GB/s and the store throughput rising from 210.76GB/s to 123.41GB/s. These changes indicate that the kernel is spending less time in shared memory and more time performing efficient operations involving global memory and registers.  

---

## **5. Experience & Conclusion**
### **5.1 What have I learned from this homework?**
  Through this project, I learned about **FlashAttention** and how to implement it using **CUDA**. I explored various optimization techniques, such as **shared memory**, **memory coalescing**, and **avoiding bank conflicts**, to improve performance.  

  However, unlike the **Blocked Floyd-Warshall Algorithm**, the optimization of **FlashAttention** seems to rely more heavily on understanding the algorithm and fine-tuning the thread and block sizes. I believe I still have some blind spots in this area and need to work on improving my understanding.  

  I also learned that when using `nvprof` for profiling, having a higher throughput, such as **Shared Memory Load Throughput**, does not necessarily result in better performance. This experience has given me a deeper understanding of GPU programming and how to optimize algorithms for parallel computing.