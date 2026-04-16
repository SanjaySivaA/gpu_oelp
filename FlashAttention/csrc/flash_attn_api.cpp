#include <torch/extension.h>
#include "kernels/flash_attn_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_fp32", &forward_fp32, "Custom Flash Attention Forward Pass (FP32)");
    m.def("forward_fp16", &forward_fp16, "Custom Flash Attention Forward Pass (FP16 Tensor Cores)");
}