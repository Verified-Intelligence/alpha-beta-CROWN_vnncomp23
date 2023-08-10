import torch

def customized_gtrsb_adv_example_finalizer(model_ori, x, best_deltas, data_max, data_min, C_mat, rhs_mat, cond_mat):
    # x and best_deltas has shape (batch, c, h, w).
    # data_min and data_max have shape (batch, spec, c, h, w).
    attack_image = torch.max(torch.min((x + best_deltas).unsqueeze(1), data_max), data_min)
    assert (attack_image >= data_min).all()
    assert (attack_image <= data_max).all()

    attack_output = model_ori(attack_image[:, 0, :]).repeat(1, attack_image.shape[1], 1)
    # [batch_size, num_or_spec, out_dim]

    # only print out the first two random start outputs of the first two examples.
    print("Adv example prediction (first 2 examples and 2 restarts):\n", attack_output[:2,:2])

    attack_output_repeat = attack_output.unsqueeze(1).repeat_interleave(torch.tensor(cond_mat[0], device=x.device), dim=2)
    # [num_example, num_restarts, num_spec, num_output]

    C_mat = C_mat.view(C_mat.shape[0], 1, -1, C_mat.shape[-1])
    # [num_example, 1, num_spec, num_output]
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], 1, -1)
    # [num_example, 1, num_spec]

    attack_margin = (C_mat * attack_output_repeat).sum(-1) - rhs_mat
    # [num_example, num_restarts, num_spec]

    print("PGD attack margin (first 2 examles and 10 specs):\n", attack_margin[:2, :, :10])
    print("number of violation: ", (attack_margin < 0).sum().item())
    # print the first 10 specifications for the first 2 examples

    return attack_image, attack_output, attack_margin
