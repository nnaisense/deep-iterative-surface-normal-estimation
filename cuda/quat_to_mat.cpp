#include <torch/torch.h>

#define IS_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

at::Tensor quat_to_mat_fw_cuda(at::Tensor x);
at::Tensor quat_to_mat_bw_cuda(at::Tensor x, at::Tensor g_y);

at::Tensor quat_to_mat_fw(at::Tensor x) {
  CHECK_INPUT(x);
  return quat_to_mat_fw_cuda(x);
}

at::Tensor quat_to_mat_bw(at::Tensor x, at::Tensor g_y) {
  CHECK_INPUT(x);
  CHECK_INPUT(g_y);
  return quat_to_mat_bw_cuda(x, g_y);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quat_to_mat_fw", &quat_to_mat_fw, "Quaternion To Rotation Matrix forward (CUDA)");
  m.def("quat_to_mat_bw", &quat_to_mat_bw, "Quaternion To Rotation Matrix backward (CUDA)");
}
