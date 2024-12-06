import unittest
from math import exp
import torch
import torch.nn.functional as F


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert (
        q.shape == k.shape == v.shape
    ), f"shape mismatch: {q.shape=} {k.shape=} {v.shape=}"
    assert len(q.shape) == 2, f"shape mismatch: {q.shape=}"

    return F.scaled_dot_product_attention(q, k, v, scale=1.0)


def naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == y.shape[0], f"shape mismatch: {x.shape} @ {y.shape}"
        z = torch.zeros(x.shape[0], y.shape[1], dtype=x.dtype)

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                for k in range(x.shape[1]):
                    z[i, j] += x[i, k] * y[k, j]

        return z

    def softmax(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2, f"shape mismatch: {x.shape=}"

        # safe softmax
        # 1st pass: find max value in each row
        m = torch.full((x.shape[0],), float("-inf"), dtype=x.dtype)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                m[i] = max(x[i, j].item(), m[i].item())
        # 2nd pass: sum of exp(x[i, :] - m[i]) for each row
        d = torch.zeros_like(m)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                d[i] += exp(x[i, j] - m[i])
        # 3rd pass: exp(x[i, j] - m[i]) / d[i]
        y = torch.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y[i, j] = exp(x[i, j] - m[i]) / d[i]

        return y

    x = matmul(q, k.T)
    a = softmax(x)
    o = matmul(a, v)
    return o


def flash(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    M, D = query.shape
    N, _ = key.shape
    assert key.shape[1] == D, f"{key.shape=}"
    assert value.shape == (N, D), f"{value.shape=}"

    out = torch.zeros((M, D), dtype=query.dtype)

    for k in range(M):
        q_ = query[k, :]  # k-th row of query
        assert len(q_.shape) == 1, f"{q_.shape=}"
        mi_ = float("-inf")  # m_{i-1}: max for softmax
        di_ = 0.0  # d'_{i-1}denominator for softmax
        for i in range(N):
            # fetch i-th row of key
            k_ = key[i, :]
            assert len(k_.shape) == 1, f"{k_.shape=}"
            # compute Q @ K.T
            xi = torch.matmul(q_, k_).item()
            # surrogate max
            mi = max(xi, mi_)
            # surrogate sum
            exp_mi = exp(mi_ - mi)
            exp_xi = exp(xi - mi)
            di = di_ * exp_mi + exp_xi
            # surrogate output
            out[k, :] = out[k, :] * di_ * exp_mi / di + exp_xi * value[i, :] / di
            # for next iteration
            mi_ = mi
            di_ = di

    return out


class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        self.shapes = [
            (2, 3),
            (16, 32),
        ]

        self.dtypes = [
            torch.float32,
        ]

        self.inputs = []
        for shape in self.shapes:
            for dtype in self.dtypes:
                q = torch.rand(shape, dtype=dtype)
                k = torch.rand(shape, dtype=dtype)
                v = torch.rand(shape, dtype=dtype)
                self.inputs.append((q, k, v))

    def test_naive(self):
        for q, k, v in self.inputs:
            tgt = naive(q, k, v)
            ref = sdpa(q, k, v)
            self.assertTrue(
                torch.allclose(tgt, ref, atol=1e-3, rtol=1e-3), f"{tgt=} {ref=}"
            )

    def test_flash(self):
        for q, k, v in self.inputs:
            tgt = flash(q, k, v)
            ref = sdpa(q, k, v)
            self.assertTrue(
                torch.allclose(tgt, ref, atol=1e-3, rtol=1e-3), f"{tgt=} {ref=}"
            )


if __name__ == "__main__":
    import pytest

    pytest.main(["-sv", __file__])
