// volleyhub_native_debayer / debayer.cpp
// pybind11 entry point. Dispatches into the CUDA kernel.

#include <torch/extension.h>


// Forward declaration from debayer_cuda.cu
torch::Tensor bilinear_demosaic_cuda(
    torch::Tensor raw,
    int64_t pattern_idx,
    int64_t output_format_idx,
    bool normalize);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VolleyHub native CUDA debayer (bilinear interpolation, 4 Bayer patterns)";
    m.def(
        "bilinear_demosaic",
        &bilinear_demosaic_cuda,
        "Bilinear Bayer demosaic on CUDA",
        pybind11::arg("raw"),
        pybind11::arg("pattern_idx"),
        pybind11::arg("output_format_idx"),
        pybind11::arg("normalize")
    );
}
