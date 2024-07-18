import torch
import torch.nn as nn
from cpm_kernels.kernels import gemm_int8
from bminf.kernels import gemm_calc_scale, gemm_round, gemm_scale, gemm_scale_round


class W8A16LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) dtype
            scale:  (out_features,)             dtype
        Returns:
            out:    (..., out_features)
        """
        ctx.inp_shape = inp.size()
        ctx.weight_shape = weight.size()
        out_features = weight.size(0)
        inp = inp.contiguous().view(-1, weight.size(1))
        weight_from_quant = quant_w.to(torch.half) * scale_w[:, None]
        output = inp.mm(weight_from_quant.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight_from_quant = quant_w.to(torch.half) * scale_w[:, None]
        grad_output = grad_output.contiguous().view(-1, weight_from_quant.size(0))  # (..., out_features)
        grad_input = grad_output.mm(weight_from_quant)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


class W8A8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) dtype
            scale:  (out_features,)             dtype
        Returns:
            out:    (..., out_features)
        """
        ctx.inp_shape = inp.size()
        ctx.weight_shape = weight.size()
        x = inp.contiguous().view(-1, weight.size(1))
        scale_x = gemm_calc_scale(x)  # (...,)
        quant_x = gemm_round(x, scale_x)  # (..., in_features)    int8

        M = quant_x.size(0)
        K = quant_x.size(1)
        N = quant_w.size(0)

        with torch.cuda.device(inp.device):
            out = torch.empty(M, N, dtype=torch.int32, device="cuda")  # (..., out_features)
            gemm_int8(
                N, K, M,
                1, 1,
                True, False,
                quant_w.data_ptr(), quant_x.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
        out = gemm_scale(out, scale_x, scale_w, inp.dtype)

        ctx.save_for_backward(quant_x, scale_x, quant_w, scale_w)
        return out.view(*(ctx.inp_shape[:-1] + (N,)))

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        quant_x, scale_x, quant_w, scale_w = ctx.saved_tensors
        grad_f = grad_f.contiguous().view(-1, quant_w.size(0))  # (..., out_features)

        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f, scale_w)

        grad_input = torch.empty(grad_f.size(0), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_w.size(1), quant_w.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_w.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_input.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_input = gemm_scale(grad_input, scale_grad_f, None, grad_f.dtype)

        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f.transpose(0, 1).contiguous(), scale_x)

        grad_weight = torch.empty(grad_f.size(1), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_x.size(1), quant_x.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_x.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_weight.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_weight = gemm_scale(grad_weight, scale_grad_f, None, grad_f.dtype)

        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


class W8A8CLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, lambada: torch.Tensor, weight: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) dtype
            scale:  (out_features,)             dtype
        Returns:
            out:    (..., out_features)
        """
        ctx.inp_shape = inp.size()
        ctx.weight_shape = weight.size()
        x = inp.contiguous().view(-1, weight.size(1))

        # clip
        # scale_x = gemm_calc_scale(x) * lambada  # (...,)
        scale_x = x.abs().mean(dim=-1) * lambada / 127
        n_alpha = scale_x[:, None] * 127
        quant_x = torch.max(torch.min(x, n_alpha), -n_alpha)
        quant_x = gemm_round(quant_x, scale_x)  # (..., in_features)    int8

        M = quant_x.size(0)
        K = quant_x.size(1)
        N = quant_w.size(0)

        with torch.cuda.device(inp.device):
            out = torch.empty(M, N, dtype=torch.int32, device="cuda")  # (..., out_features)
            gemm_int8(
                N, K, M,
                1, 1,
                True, False,
                quant_w.data_ptr(), quant_x.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
        out = gemm_scale(out, scale_x, scale_w, inp.dtype)

        ctx.save_for_backward(x, quant_x, scale_x, quant_w, scale_w, lambada)
        return out.view(*(ctx.inp_shape[:-1] + (N,)))

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        x, quant_x, scale_x, quant_w, scale_w, lambada = ctx.saved_tensors
        grad_f = grad_f.contiguous().view(-1, quant_w.size(0))  # (..., out_features)

        # Grad input
        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f, scale_w)
        grad_input = torch.empty(grad_f.size(0), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_w.size(1), quant_w.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_w.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_input.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_input = gemm_scale(grad_input, scale_grad_f, None, grad_f.dtype)

        # Grad weight
        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f.transpose(0, 1).contiguous(), scale_x)
        grad_weight = torch.empty(grad_f.size(1), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_x.size(1), quant_x.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_x.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_weight.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_weight = gemm_scale(grad_weight, scale_grad_f, None, grad_f.dtype)

        # Grad lambda
        scale_x = scale_x[:, None]
        n_alpha = scale_x * 127
        not_clipped = torch.logical_and(x < n_alpha, x > -n_alpha)
        grad_lambada = (grad_input * (quant_x.to(grad_input.dtype) * scale_x - not_clipped * x) / lambada).sum()

        return grad_input.view(ctx.inp_shape), grad_lambada, grad_weight.view(ctx.weight_shape), None, None

class W8A8ALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) dtype
            scale:  (out_features,)             dtype
        Returns:
            out:    (..., out_features)
        """
        ctx.inp_shape = inp.size()
        ctx.weight_shape = weight.size()
        x = inp.contiguous().view(-1, weight.size(1))

        x_min = x.min(dim=-1).values
        x_max = x.max(dim=-1).values
        scale_x = (x_max - x_min) / 254
        zero_x = (torch.round(x_min / scale_x) + 127).to(torch.int8)
        quant_x = (torch.round(x / scale_x[:, None]) - zero_x[:, None]).to(torch.int8)

        M = quant_x.size(0)
        K = quant_x.size(1)
        N = quant_w.size(0)

        with torch.cuda.device(inp.device):
            out = torch.empty(M, N, dtype=torch.int32, device="cuda")  # (..., out_features)
            gemm_int8(
                N, K, M,
                1, 1,
                True, False,
                quant_w.data_ptr(), quant_x.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
            out2 = quant_w.sum(dim=-1, dtype=torch.int32).view(1, -1).expand(M, N) * zero_x[:, None]
        out = gemm_scale(out + out2, scale_x, scale_w, inp.dtype)

        ctx.save_for_backward(x, quant_w, scale_w)
        return out.view(*(ctx.inp_shape[:-1] + (N,)))

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        x, quant_w, scale_w = ctx.saved_tensors
        grad_f = grad_f.contiguous().view(-1, quant_w.size(0))  # (..., out_features)

        quant_grad_f, scale_grad_f = gemm_scale_round(grad_f, scale_w)

        grad_input = torch.empty(grad_f.size(0), quant_w.size(1), dtype=torch.int32, device="cuda")
        gemm_int8(
            quant_w.size(1), quant_w.size(0), quant_grad_f.size(0),
            1, 1,
            True, False,
            # Here we need a handwritten transpose to enable TensorCore
            quant_w.transpose(0, 1).contiguous().data_ptr(), quant_grad_f.data_ptr(),
            grad_input.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        grad_input = gemm_scale(grad_input, scale_grad_f, None, grad_f.dtype)

        grad_weight = grad_f.t().mm(x)

        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None