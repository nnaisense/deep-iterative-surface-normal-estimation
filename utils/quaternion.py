import torch

if torch.cuda.is_available():
    import quat_to_mat

class QuatToMat(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        assert torch.cuda.is_available()
        assert x.size(-1) == 4
        self.save_for_backward(x)
        return quat_to_mat.quat_to_mat_fw(x)

    @staticmethod
    def backward(self, g_y):
        x, = self.saved_variables
        g_x = None

        if self.needs_input_grad[0]:
            g_x = quat_to_mat.quat_to_mat_bw(x, g_y)

        return g_x

