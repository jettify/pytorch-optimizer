"""Shampoo must raise each preconditioner to -1/(2k) (gh issue 502).

Gupta, Koren, Singer 2018 prescribe L^{-1/4} G R^{-1/4} for matrices
(order 2) and L^{-1/(2k)} per mode factor for order-k tensors. The code
used -1/order, i.e. twice the paper exponent.
"""
import torch
import torch_optimizer as optim


def _eigh_power(matrix, power):
    w, v = torch.linalg.eigh(matrix)
    return v @ torch.diag(w.clamp_min(1e-12).pow(power)) @ v.t()


def test_matrix_preconditioner_is_inverse_fourth_root():
    # X = M^{-1/4} iff X^4 = M^{-1}. No SVD-vs-eigh sensitivity.
    torch.manual_seed(502)
    p = torch.nn.Parameter(torch.randn(4, 3))
    opt = optim.Shampoo([p], lr=1e-3, momentum=0, update_freq=1)
    p.grad = torch.randn(4, 3)
    opt.step()
    state = opt.state[p]
    for dim_id in range(2):
        precond = state["precond_{}".format(dim_id)].cpu()
        inv = state["inv_precond_{}".format(dim_id)].cpu()
        fourth = inv @ inv @ inv @ inv
        assert torch.allclose(
            fourth, torch.inverse(precond.double()).float(),
            rtol=1e-2, atol=1e-2,
        )


def test_vector_preconditioner_is_inverse_sqrt():
    # X = M^{-1/2} iff X^2 = M^{-1}.
    torch.manual_seed(502)
    p = torch.nn.Parameter(torch.randn(5))
    opt = optim.Shampoo([p], lr=1e-3, momentum=0, update_freq=1)
    p.grad = torch.randn(5)
    opt.step()
    state = opt.state[p]
    precond = state["precond_0"].cpu()
    inv = state["inv_precond_0"].cpu()
    # eps*I keeps one eigenvalue near 1e-4, so the inverse entries reach
    # 1e4 and the SVD power carries ~1% relative error.
    assert torch.allclose(
        inv @ inv, torch.inverse(precond.double()).float(),
        rtol=5e-2, atol=1.0,
    )


def test_paper_exponent_beats_old_exponent_on_least_squares():
    # Same setup, same lr: the paper exponent (-1/4 per side) must reach a
    # lower loss than the old exponent (-1/2 per side).
    torch.manual_seed(7)
    A = torch.randn(20, 6)
    b = torch.randn(20)

    def run(power, steps=60, lr=0.05):
        x = torch.zeros(6)
        L = torch.eye(6) * 1e-4
        R = torch.eye(1) * 1e-4
        for _ in range(steps):
            g = A.T @ (A @ x - b)
            G = g.unsqueeze(1)
            L = L + G @ G.t()
            R = R + G.t() @ G
            upd = _eigh_power(L, power) @ G @ _eigh_power(R, power)
            x = x - lr * upd.squeeze(1)
        return ((A @ x - b) ** 2).mean().item()

    assert run(-0.25) < run(-0.5)
