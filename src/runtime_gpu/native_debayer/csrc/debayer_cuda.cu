// volleyhub_native_debayer / debayer_cuda.cu
//
// Bilinear-interpolation Bayer demosaic on CUDA. Supports four Bayer
// patterns and writes a (B, 3, H, W) float buffer in either RGB or BGR
// channel order, with optional /255 normalization.
//
// Boundary handling: clamp-to-edge so the kernel never reads out of
// bounds. This matches what cv2.cvtColor does on Bayer reflect borders
// closely enough for our benchmarking and downstream YOLO inference.
//
// Pattern indexing (matches Python adapter):
//   0 = RG  : R at (0,0), G at (0,1)+(1,0), B at (1,1)
//   1 = BG  : B at (0,0), G at (0,1)+(1,0), R at (1,1)
//   2 = GR  : G at (0,0)+(1,1), R at (0,1), B at (1,0)
//   3 = GB  : G at (0,0)+(1,1), B at (0,1), R at (1,0)

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


namespace {

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename scalar_in_t>
__global__ void bilinear_demosaic_kernel(
    const scalar_in_t* __restrict__ raw,
    float* __restrict__ out,
    int B, int H, int W,
    int pattern_idx,
    int output_format_idx,
    int normalize)
{
    const int total = B * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int b  = idx / (H * W);
    const int rem = idx - b * (H * W);
    const int y  = rem / W;
    const int x  = rem - y * W;

    const int ym = clamp_int(y - 1, 0, H - 1);
    const int yp = clamp_int(y + 1, 0, H - 1);
    const int xm = clamp_int(x - 1, 0, W - 1);
    const int xp = clamp_int(x + 1, 0, W - 1);

    const int base = b * H * W;

    // Position of R within the 2x2 mosaic cell, parametrized by pattern.
    int R_py, R_px;
    switch (pattern_idx) {
        case 0: R_py = 0; R_px = 0; break;  // RG
        case 1: R_py = 1; R_px = 1; break;  // BG
        case 2: R_py = 0; R_px = 1; break;  // GR
        case 3: R_py = 1; R_px = 0; break;  // GB
        default: R_py = 0; R_px = 0; break;
    }
    const int B_py = 1 - R_py;
    const int B_px = 1 - R_px;

    const int py = y & 1;
    const int px = x & 1;

    // Inline neighbour reads. Macro keeps the kernel readable while
    // staying compatible with default nvcc (no --extended-lambda).
    #define V_AT(yy, xx) (static_cast<float>(raw[base + (yy) * W + (xx)]))

    float R, G, B_;
    if (py == R_py && px == R_px) {
        // R pixel
        R  = V_AT(y, x);
        G  = (V_AT(ym, x) + V_AT(yp, x) + V_AT(y, xm) + V_AT(y, xp)) * 0.25f;
        B_ = (V_AT(ym, xm) + V_AT(ym, xp) + V_AT(yp, xm) + V_AT(yp, xp)) * 0.25f;
    } else if (py == B_py && px == B_px) {
        // B pixel
        B_ = V_AT(y, x);
        G  = (V_AT(ym, x) + V_AT(yp, x) + V_AT(y, xm) + V_AT(y, xp)) * 0.25f;
        R  = (V_AT(ym, xm) + V_AT(ym, xp) + V_AT(yp, xm) + V_AT(yp, xp)) * 0.25f;
    } else {
        // G pixel — R/B alternation depends on row parity
        G = V_AT(y, x);
        if (py == R_py) {
            R  = (V_AT(y, xm) + V_AT(y, xp)) * 0.5f;
            B_ = (V_AT(ym, x) + V_AT(yp, x)) * 0.5f;
        } else {
            B_ = (V_AT(y, xm) + V_AT(y, xp)) * 0.5f;
            R  = (V_AT(ym, x) + V_AT(yp, x)) * 0.5f;
        }
    }
    #undef V_AT

    float ch0, ch1, ch2;
    if (output_format_idx == 0) {  // RGB
        ch0 = R; ch1 = G; ch2 = B_;
    } else {                       // BGR
        ch0 = B_; ch1 = G; ch2 = R;
    }

    if (normalize) {
        const float inv = 1.0f / 255.0f;
        ch0 *= inv; ch1 *= inv; ch2 *= inv;
    }

    const int plane = H * W;
    const int batch_offset = b * 3 * plane;
    out[batch_offset + 0 * plane + y * W + x] = ch0;
    out[batch_offset + 1 * plane + y * W + x] = ch1;
    out[batch_offset + 2 * plane + y * W + x] = ch2;
}

} // anonymous namespace


torch::Tensor bilinear_demosaic_cuda(
    torch::Tensor raw,
    int64_t pattern_idx,
    int64_t output_format_idx,
    bool normalize)
{
    TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
    TORCH_CHECK(raw.dim() == 3, "raw must be (B, H, W)");
    TORCH_CHECK(raw.scalar_type() == torch::kUInt8 || raw.scalar_type() == torch::kInt32,
                "raw must be uint8 (uint16 not yet supported on this build)");

    const at::cuda::CUDAGuard device_guard(raw.device());

    const int64_t B = raw.size(0);
    const int64_t H = raw.size(1);
    const int64_t W = raw.size(2);

    raw = raw.contiguous();

    auto opts = torch::TensorOptions()
                    .device(raw.device())
                    .dtype(torch::kFloat32);
    auto out = torch::empty({B, 3, H, W}, opts);

    const int64_t total = B * H * W;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (raw.scalar_type() == torch::kUInt8) {
        bilinear_demosaic_kernel<uint8_t><<<blocks, threads, 0, stream.stream()>>>(
            raw.data_ptr<uint8_t>(),
            out.data_ptr<float>(),
            static_cast<int>(B), static_cast<int>(H), static_cast<int>(W),
            static_cast<int>(pattern_idx),
            static_cast<int>(output_format_idx),
            normalize ? 1 : 0);
    }

    return out;
}
