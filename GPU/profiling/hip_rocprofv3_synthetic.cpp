#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <roctx.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if(_e != hipSuccess) {                                                  \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << hipGetErrorString(_e) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while(0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status _s = (call);                                             \
        if(_s != rocblas_status_success) {                                      \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__  \
                      << " -> status " << static_cast<int>(_s) << std::endl;  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while(0)

__global__ void saxpy_kernel(float* y, const float* x, float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = a * x[i] + y[i];
}

__global__ void busy_wait_kernel(float* data, int n, int inner_iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        float v = data[i];
        for(int k = 0; k < inner_iters; ++k) {
            v = sinf(v) + cosf(v);
        }
        data[i] = v;
    }
}

int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::stoi(argv[1]) : (1 << 20);
    const int gemm_m = 512, gemm_n = 512, gemm_k = 512;
    const int reps = (argc > 2) ? std::stoi(argv[2]) : 5;

    std::cout << "Synthetic profiling app start (n=" << n << ", reps=" << reps << ")\n";

    roctxRangePush("setup");
    HIP_CHECK(hipSetDevice(0));

    hipStream_t stream_a{}, stream_b{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream_a, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_b, hipStreamNonBlocking));

    hipEvent_t evt_copy_done{}, evt_kernel_done{};
    HIP_CHECK(hipEventCreate(&evt_copy_done));
    HIP_CHECK(hipEventCreate(&evt_kernel_done));

    std::vector<float> h_x(n, 1.0f), h_y(n, 2.0f), h_out(n, 0.0f);
    float *d_x = nullptr, *d_y = nullptr, *d_tmp = nullptr;
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tmp, n * sizeof(float)));

    const size_t a_size = gemm_m * gemm_k;
    const size_t b_size = gemm_k * gemm_n;
    const size_t c_size = gemm_m * gemm_n;

    std::vector<float> h_A(a_size, 0.01f), h_B(b_size, 0.02f), h_C(c_size, 0.0f);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, a_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, b_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, c_size * sizeof(float)));

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream_b));
    roctxRangePop();

    const float alpha = 1.1f;
    const float beta = 0.9f;

    for(int iter = 0; iter < reps; ++iter) {
        roctxRangePush("iteration");

        roctxRangePush("h2d-async-copy");
        HIP_CHECK(hipMemcpyAsync(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice, stream_a));
        HIP_CHECK(hipMemcpyAsync(d_y, h_y.data(), n * sizeof(float), hipMemcpyHostToDevice, stream_a));
        HIP_CHECK(hipMemcpyAsync(d_A, h_A.data(), a_size * sizeof(float), hipMemcpyHostToDevice, stream_b));
        HIP_CHECK(hipMemcpyAsync(d_B, h_B.data(), b_size * sizeof(float), hipMemcpyHostToDevice, stream_b));
        HIP_CHECK(hipEventRecord(evt_copy_done, stream_a));
        roctxRangePop();

        roctxRangePush("stream-sync-and-kernel");
        HIP_CHECK(hipStreamWaitEvent(stream_b, evt_copy_done, 0));
        int block = 256;
        int grid = (n + block - 1) / block;
        hipLaunchKernelGGL(saxpy_kernel, dim3(grid), dim3(block), 0, stream_a, d_y, d_x, 2.5f, n);
        hipLaunchKernelGGL(busy_wait_kernel, dim3(grid), dim3(block), 0, stream_a, d_y, n, 100);
        HIP_CHECK(hipEventRecord(evt_kernel_done, stream_a));
        roctxRangePop();

        roctxRangePush("d2d-copy-and-rocblas");
        HIP_CHECK(hipStreamWaitEvent(stream_b, evt_kernel_done, 0));
        HIP_CHECK(hipMemcpyAsync(d_tmp, d_y, n * sizeof(float), hipMemcpyDeviceToDevice, stream_b));

        ROCBLAS_CHECK(rocblas_sgemm(handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    gemm_m,
                                    gemm_n,
                                    gemm_k,
                                    &alpha,
                                    d_A,
                                    gemm_m,
                                    d_B,
                                    gemm_k,
                                    &beta,
                                    d_C,
                                    gemm_m));
        roctxRangePop();

        roctxRangePush("sync-and-d2h");
        HIP_CHECK(hipStreamSynchronize(stream_a));
        HIP_CHECK(hipMemcpyAsync(h_out.data(), d_tmp, n * sizeof(float), hipMemcpyDeviceToHost, stream_b));
        HIP_CHECK(hipMemcpyAsync(h_C.data(), d_C, c_size * sizeof(float), hipMemcpyDeviceToHost, stream_b));
        HIP_CHECK(hipDeviceSynchronize());
        roctxRangePop();

        roctxRangePop();
    }

    double checksum = 0.0;
    for(int i = 0; i < n; i += (n / 32 + 1)) checksum += h_out[i];
    for(int i = 0; i < static_cast<int>(c_size); i += (c_size / 32 + 1)) checksum += h_C[i];

    std::cout << "Done. checksum=" << checksum << "\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(evt_copy_done));
    HIP_CHECK(hipEventDestroy(evt_kernel_done));
    HIP_CHECK(hipStreamDestroy(stream_a));
    HIP_CHECK(hipStreamDestroy(stream_b));

    return 0;
}
