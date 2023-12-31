Bound Options
====================

Bound options can be set by passing a dictionary to the `bound_opts` argument for `BoundedModule`.
This page lists available bound options.

## Arguments for Optimizing Bounds (`optimize_bound_args`)

Arguments for optimizing bounds with the `CROWN-Optimized` method can be provided as a dictionary. Available arguments include:

* `enable_alpha_crown` (bool, default `True`): Enable α-CROWN (optimized CROWN/LiRPA).

* `enable_beta_crown` (bool, default `False`): Enable β-CROWN.

* `optimizer` (str, default `adam`): Optimzier. Set it to `adam-autolr` to use `AdamElementLR`, or `sgd` to use SGD.

* `lr_alpha` (float, default 0.5), `lr_beta` (default 0.05): Learning rates for α and β parameters in α-CROWN and β-CROWN.

* `lr_decay` (float, default 0.98): Learning rate decay factor for the `ExponentialLR` scheduler.

* `iteration` (int): Number of optimization iterations.

* `loss_reduction_func` (function): Function for loss reduction over the specification dimension. By default, use `auto_LiRPA.utils.reduction_sum` which sumes the bound over all batch elements and specifications.

* `stop_criterion_func` (function): Function for the criterion of stopping optimization early; it returns a tensor of `torch.bool` with `batch_size` elements. By default, it is a lambda function that always returns `False` . Several pre-defined options are `auto_LiRPA.utils.stop_criterion_min`, `auto_LiRPA.utils.stop_criterion_mean`, `auto_LiRPA.utils.stop_criterion_max` and `auto_LiRPA.utils.stop_criterion_sum`. For example, `auto_LiRPA.utils.stop_criterion_min` checks the minimum bound over all specifications of a batch element and returns `True` for that element when the minimum bound is greater than a specified threshold.

* `keep_best` (bool, default `True`): If `True`, save α, β and bounds at the best iteration. Otherwise the last iteration result is used.

* `use_shared_alpha` (bool, default `False`): If `True`, all intermediate neurons from the same layer share the same set of α variables during bound optimization. For a very large model, enabling this option can save memory, at a cost of slightly looser bound.

* `fix_intermediate_layer_bounds` (bool, default `True`): Only optimize bounds of last layer during alpha/beta CROWN.

* `init_alpha` (bool, default `True`): Initial alpha variables by calling CROWN once.

* `early_stop_patience` (int, default, 10): Number of iterations that we will start considering early stop if tracking no improvement.

* `start_save_best` (float, default 0.5): Start to save optimized best bounds when current_iteration > int(iteration*start_save_best)


## ReLU (`relu`):

There are different choices for the lower bound relaxation of unstable ReLU activations (see the [CROWN paper](https://arxiv.org/pdf/1811.00866.pdf)):

* `adaptive` (default): For unstable neurons, when the slope of the upper bound is greater than one, use 1 as the slope of the lower bound, otherwise use 0 as the slope of the lower bound (this is described as CROWN-Ada in the original CROWN paper). Please also use this option if the `CROWN-Optimized` bound is used and the lower bound needs to be optimized.

* `same-slope`: Make the slope for lower bound the same as the upper bound.

* `zero-lb`: Always use 0 as the slope of lower bound for unstable neurons.

* `one-lb`: Always use 1 as the slope of lower bound for unstable neurons.

* `reversed-adaptive`: For unstable neurons, when the slope of the upper bound is greater than one, use 0 as the slope of the lower bound, otherwise use 1 as the slope of the lower bound.

## Other Options

* `loss_fusion`: If `True`, this bounded module has loss fusion, i.e., the loss function is also included in the module and the output of the model is the loss rather than logits.

* `deterministic`: If `True`, make PyTorch use deterministic algorithms.

* `matmul`: If set to `economic`, use a memory-efficient IBP implementation for relaxing the `matmul` operation when both arguments of `matmul` are perturbed, which does not expand all the elementary multiplications to save memory.
