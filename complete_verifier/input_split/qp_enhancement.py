"""Experimental code with QP.

Hard-coded for the output feedback model only.
"""

import time
import torch


@torch.no_grad()
def run_qp(net, new_x, thresholds, stop_criterion_func, lb):
    unverified = torch.logical_not(
        stop_criterion_func(thresholds)(lb).any(dim=-1))
    weight = net['/257'].value
    rhs = thresholds[0][1]
    x_L_ = new_x.ptb.x_L[unverified]
    x_U_ = new_x.ptb.x_U[unverified]
    start_time = time.time()
    print('Solving QP')
    ret_qp = solve_qph(x_L_, x_U_, weight)
    print('Time:', time.time() - start_time)
    print('Verified by QP:', (ret_qp > rhs).sum())
    lb[unverified, 1] = ret_qp
    return lb


def solve_qph(x_L, x_U, w):
  # qpth is an optional dependency right now
  from qpth.qp import QPFunction

  # Q_, p_, G_, h_, A_, b_
  # \hat z =   argmin_z 1/2 z^T Q z + p^T z
  #                 subject to Gz <= h
  #                             Az  = b

  #     where Q \in S^{nz,nz},
  #         S^{nz,nz} is the set of all positive semi-definite matrices,
  #         p \in R^{nz}
  #         G \in R^{nineq,nz}
  #         h \in R^{nineq}
  #         A \in R^{neq,nz}
  #         b \in R^{neq}
  nz = x_L.shape[-1]
  device = x_L.device
  x = QPFunction(verbose=-1)(
    w,
    torch.zeros(nz, device=device),
    torch.concat([-torch.eye(nz, device=device),
                  torch.eye(nz, device=device)], dim=0),
    torch.concat([-x_L, x_U], dim=-1),
    torch.zeros(0, nz, device=device),
    torch.zeros(0, device=device),
  )
  ret_qph = (x*x.matmul(w)).sum(dim=-1)
  return ret_qph
