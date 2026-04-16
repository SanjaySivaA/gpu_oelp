#pragma once
#include <torch/extension.h>

// Forward declarations of our wrapper functions
torch::Tensor forward_fp32(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor forward_fp16(torch::Tensor Q, torch::Tensor K, torch::Tensor V);