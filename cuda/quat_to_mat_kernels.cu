#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void quat_to_mat_fw_kernel(const scalar_t *__restrict__ x, scalar_t *y,
                                 size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    auto j = i * 4, k = i * 9;

    scalar_t x0 = x[j + 0], x1 = x[j + 1], x2 = x[j + 2], x3 = x[j + 3];

    y[k + 0] = 1 - 2 * (x2 * x2 + x3 * x3);
    y[k + 1] = 2 * (x1 * x2 - x0 * x3);
    y[k + 2] = 2 * (x1 * x3 + x0 * x2);
    y[k + 3] = 2 * (x1 * x2 + x0 * x3);
    y[k + 4] = 1 - 2 * (x1 * x1 + x3 * x3);
    y[k + 5] = 2 * (x2 * x3 - x0 * x1);
    y[k + 6] = 2 * (x1 * x3 - x0 * x2);
    y[k + 7] = 2 * (x2 * x3 + x0 * x1);
    y[k + 8] = 1 - 2 * (x1 * x1 + x2 * x2);
  }
}

at::Tensor quat_to_mat_fw_cuda(at::Tensor x) {
  std::vector<int64_t> size(x.sizes().begin(), x.sizes().end());
  size.back() = 3;
  size.push_back(3);
  auto y = at::empty(size, x.options());
  size_t n = x.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES(x.type(), "quat_to_mat_fw_kernel", [&] {
    quat_to_mat_fw_kernel<scalar_t>
        <<<BLOCKS(n), THREADS>>>(x.data<scalar_t>(), y.data<scalar_t>(), n);
  });

  return y;
}

template <typename scalar_t>
__global__ void quat_to_mat_bw_kernel(const scalar_t *__restrict__ x,
                                 const scalar_t *__restrict__ g_y,
                                 scalar_t *g_x, size_t numel) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = idx; i < numel; i += stride) {
    auto j = i * 4, k = i * 9;
    scalar_t x0 = x[j + 0], x1 = x[j + 1], x2 = x[j + 2], x3 = x[j + 3];
    scalar_t g_y0 = g_y[k + 0], g_y1 = g_y[k + 1], g_y2 = g_y[k + 2],
             g_y3 = g_y[k + 3], g_y4 = g_y[k + 4], g_y5 = g_y[k + 5],
             g_y6 = g_y[k + 6], g_y7 = g_y[k + 7], g_y8 = g_y[k + 8];

    g_x[j + 0] = 2 * (-x3 * g_y1 + x2 * g_y2 + x3 * g_y3 - x1 * g_y5 -
                      x2 * g_y6 + x1 * g_y7);
    g_x[j + 1] = 2 * (x2 * g_y1 + x3 * g_y2 + x2 * g_y3 - 2 * x1 * g_y4 -
                      x0 * g_y5 + x3 * g_y6 + x0 * g_y7 - 2 * x1 * g_y8);
    g_x[j + 2] = 2 * (-2 * x2 * g_y0 + x1 * g_y1 + x0 * g_y2 + x1 * g_y3 +
                      x3 * g_y5 - x0 * g_y6 + x3 * g_y7 - 2 * x2 * g_y8);
    g_x[j + 3] = 2 * (-2 * x3 * g_y0 - x0 * g_y1 + x1 * g_y2 + x0 * g_y3 -
                      2 * x3 * g_y4 + x2 * g_y5 + x1 * g_y6 + x2 * g_y7);
  }
}

at::Tensor quat_to_mat_bw_cuda(at::Tensor x, at::Tensor g_y) {
  auto g_x = at::empty_like(x);
  size_t n = x.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES(x.type(), "quat_to_mat_bw_kernel", [&] {
    quat_to_mat_bw_kernel<scalar_t><<<BLOCKS(n), THREADS>>>(
        x.data<scalar_t>(), g_y.data<scalar_t>(), g_x.data<scalar_t>(), n);
  });

  return g_x;
}


