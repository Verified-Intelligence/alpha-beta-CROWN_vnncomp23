import torch
import torch.nn.functional as F

from .activation_base import BoundActivation
from .nonlinear import BoundOptimizableNonLinear


class BoundSin(BoundOptimizableNonLinear):
    # Lookup tables shared by all BoundSin classes.
    xl_lower_tb = None
    xl_upper_tb = None
    xu_lower_tb = None
    xu_upper_tb = None
    func, d_func = torch.sin, torch.cos
    n_table_entries = 1001

    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = True
        self.use_precompute = True
        self.act_func = torch.sin
        self.d_act_func = torch.cos

        # Bound limits used by IBP.
        self.ibp_max_point = torch.pi / 2
        self.ibp_min_point = torch.pi * 3 / 2

        self.all_table_x = torch.linspace(
            0, 2 * torch.pi, BoundSin.n_table_entries, device=self.device)
        if self.use_precompute:
            self.precompute_relaxation(self.act_func, self.d_act_func, x_limit = torch.pi / 2)
        if BoundSin.xl_lower_tb is None:
            # Generate look-up tables.
            BoundSin.xl_lower_tb = BoundSin.get_lower_left_bound(self.all_table_x)
            BoundSin.xl_upper_tb = BoundSin.get_upper_left_bound(self.all_table_x)
            BoundSin.xu_lower_tb = BoundSin.get_lower_right_bound(self.all_table_x)
            BoundSin.xu_upper_tb = BoundSin.get_upper_right_bound(self.all_table_x)

    def d2_act_func(self, x):
        return -torch.sin(x)

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = [size_spec] + list(l.shape)
        alpha = torch.empty(16, *shape, device=l.device)
        alpha.data[:4] = ((l + u) / 2).unsqueeze(0).expand(4, *shape)
        alpha.data[4:6] = self.tp_both_lower_init[name_start].expand(2, *shape)
        alpha.data[6:8] = self.tp_both_upper_init[name_start].expand(2, *shape)
        alpha.data[8:10] = self.imp_both_lower_init_0[name_start].expand(2, *shape)
        alpha.data[10:12] = self.imp_both_upper_init_0[name_start].expand(2, *shape)
        alpha.data[12:14] = self.imp_both_lower_init_1[name_start].expand(2, *shape)
        alpha.data[14:16] = self.imp_both_upper_init_1[name_start].expand(2, *shape)
        return alpha

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {}
        self.tp_both_upper_init = {}
        self.imp_both_upper_init_0 = {}
        self.imp_both_lower_init_0 = {}
        self.imp_both_upper_init_1 = {}
        self.imp_both_lower_init_1 = {}

    def generate_inflections(self, lb, ub):
        max_ele = torch.max(ub)
        min_ele = torch.min(lb)

        self.extremes = []
        k = int((min_ele - torch.pi / 2) / torch.pi)
        extreme = (k + 1.5) * torch.pi if (k + 0.5) * torch.pi < min_ele else (k + 0.5) * torch.pi
        while (extreme <= max_ele):
            self.extremes.append(extreme)
            extreme += torch.pi

        self.inflections = []
        k = int(min_ele / torch.pi)
        inflection = (k + 1) * torch.pi if k * torch.pi < min_ele else k * torch.pi
        while (inflection <= max_ele):
            self.inflections.append(inflection)
            inflection += torch.pi

    def generate_d_lower_upper(self, lower, upper):
        # Indices of neurons with input upper bound >=0, whose optimal slope to lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0, they will be masked later.
        k_tensor = (upper / (2 * torch.pi)).to(torch.long)
        k_tensor = torch.where(k_tensor < 0, k_tensor - 1, k_tensor)
        upper_clamped = upper % (2 * torch.pi)
        case1_mask = torch.logical_and(upper_clamped >= 0, upper_clamped <= torch.pi / 2)
        upper_clamped_new = upper_clamped.clamp(min=0, max=(torch.pi / 2))
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            (upper_clamped_new / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # Lookup the lower bound slope from the pre-computed table.
        d_lower = (torch.index_select(self.d_lower, 0, index).view(lower.shape) + k_tensor * 2 * torch.pi) * case1_mask

        case2_mask = torch.logical_and(upper_clamped >= torch.pi, upper_clamped <= 3 * torch.pi / 2)
        upper_clamped_new = upper_clamped.clamp(min=torch.pi, max=(3 * torch.pi / 2))
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            ((torch.pi - upper_clamped_new) / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # Lookup the lower bound slope from the pre-computed table.
        d_upper = (torch.pi - torch.index_select(self.d_upper, 0, index).view(lower.shape) + k_tensor * 2 * torch.pi) * case2_mask

        # Indices of neurons with lower bound <=0, whose optimal slope to upper bound the function was pre-computed.
        k_tensor = (lower / (2 * torch.pi)).to(torch.long)
        k_tensor = torch.where(k_tensor < 0, k_tensor - 1, k_tensor)
        lower_clamped = lower % (2 * torch.pi)
        case3_mask = torch.logical_and(lower_clamped >= 3 * torch.pi / 2, lower_clamped <= 2 * torch.pi)
        lower_clamped_new = lower_clamped.clamp(min=(3 * torch.pi / 2), max=2 * torch.pi)
        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            ((lower_clamped_new - 2 * torch.pi) / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_upper += (torch.index_select(self.d_upper, 0, index).view(upper.shape) + k_tensor * 2 * torch.pi) * case3_mask

        case4_mask = torch.logical_and(lower_clamped >= torch.pi / 2, lower_clamped <= torch.pi)
        lower_clamped_new = lower_clamped.clamp(min=(torch.pi / 2), max=3 * torch.pi)
        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            ((torch.pi - lower_clamped_new) / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_lower += (torch.pi - torch.index_select(self.d_lower, 0, index).view(upper.shape) + (k_tensor - 1) * 2 * torch.pi) * case4_mask
        return d_lower, d_upper

    @staticmethod
    def n_crossing(start, end, s):
        """Check how many times we will encounter value s + k*2*pi within start and end for any integer k."""
        cycles = torch.floor((end - start) / (2 * torch.pi))  # Number of 2pi cycles.
        # Move s and end to the same 2 * pi cycle as start.
        dist = torch.floor((s - start) / (2 * torch.pi))
        real_s = s - dist * 2 * torch.pi
        real_end = end - cycles * 2 * torch.pi
        return (real_s >= start).to(s) * (real_s <= real_end).to(s) + cycles

    @staticmethod
    def arcsin(c):
        """Arcsin with gradient fixes.

        arcsin(-1) and arcsin(1) have pathological gradients and should be avoided.
        """
        if c.min() > -1 and c.max() < 1:
            return torch.arcsin(c)
        c_ = c.clone()
        mask_neg = c == -1
        mask_pos = c == 1
        c_[mask_neg] = 0
        c_[mask_pos] = 0
        ret = torch.arcsin(c_)
        ret[mask_neg] = -torch.pi / 2
        ret[mask_pos] = torch.pi / 2
        return ret

    @staticmethod
    def get_intersection(start, end, c, theta=0.):
        """Get the number of intersections between y = sin(x + theta) and y = c between start and end."""
        # Use arcsine to find the first 2 intersections.
        crossing1 = BoundSin.arcsin(c) - theta
        crossing2 = torch.pi - crossing1 - 2 * theta  # Problematic at exact 1/2 pi, but ok in our case (happens only when lb=ub).
        return BoundSin.n_crossing(start, end, crossing1) + BoundSin.n_crossing(start, end, crossing2)

    @staticmethod
    def get_bounding_slope(xl, xu, c, theta=0.):
        """Find the point between xl and xu such that the tangent line at that point is a lower/upper bound."""
        dtype = xl.dtype
        crossing1 = BoundSin.arcsin(c) - theta  # output is [-0.5 pi, 0.5 pi] - theta. For cosine, theta=0.5 pi and crossing point is between -pi to 0.
        crossing2 = torch.pi - crossing1 - 2 * theta  # output is [0.5pi, 1.5pi] - theta. For cosine, it is between 0 and pi.
        # Find the crossing point between xl and xu.
        # First see how xl is away from the [-0.5pi, 1.5pi] range for sine or [-pi, pi] range for cosine.
        cycles1 = torch.floor((crossing1 - xl) / (2 * torch.pi)) * 2 * torch.pi
        # Move the two crossing points to the same cycle as xl.
        crossing1_moved = crossing1  - cycles1
        cycles2 = torch.floor((crossing2 - xl) / (2 * torch.pi)) * 2 * torch.pi
        crossing2_moved = crossing2 - cycles2
        # Then check which crossing point is the actual tangent point between xl and xu.
        crossing1_used = (crossing1_moved >= xl).to(dtype) * (crossing1_moved <= xu).to(dtype)
        crossing2_used = (crossing2_moved >= xl).to(dtype) * (crossing2_moved <= xu).to(dtype)
        crossing_point = crossing1_used * crossing1_moved + crossing2_used * crossing2_moved
        return crossing_point

    @staticmethod
    def check_bound(tangent_point, x):
        """Check whether the tangent line at tangent_point is a valid lower/upper bound for x."""
        # evaluate the value of the tangent line at x and see it is >= 0 or <=0.
        d = BoundSin.d_func(tangent_point)
        val = d * (x - tangent_point) + BoundSin.func(tangent_point)
        # We want a positive margin when finding a lower line, but as close to 0 as possible.
        # We want a negative margin when finding a upper line, but as close to 0 as possible.
        margin = BoundSin.func(x) - val
        return margin

    @staticmethod
    @torch.no_grad()
    def get_lower_left_bound(xl, steps=20):
        """Get a global lower bound given lower bound on x. Return slope and intercept."""
        dtype = xl.dtype
        # Constrain xl into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xl + 0.5 * torch.pi) / (2 * torch.pi)) * (2 * torch.pi)
        xl = xl - cycles
        use_tangent_line = (xl >= torch.pi).to(dtype)
        # Case 1: xl > pi, Lower tangent line is the only possible lower bound.
        case1_d = BoundSin.d_func(xl)
        case1_b = BoundSin.func(xl) - case1_d * (xl + cycles)
        # Case 2: Binary search needed. Testing from another tangent endpoint in [pi, 1.5*pi]. It must be in this region.
        left = torch.pi * torch.ones_like(xl)
        # The right end guarantees the margin > 0 because it is basically a IBP lower bound (-1).
        right = (1.5 * torch.pi) * torch.ones_like(xl)
        last_right = right.clone()
        for _ in range(steps):
            mid = (left + right) / 2.
            margin = BoundSin.check_bound(mid, xl)
            pos_mask = (margin > 0).to(dtype)  # We want to margin > 0 but at small as possible.
            neg_mask = 1.0 - pos_mask
            right = mid * pos_mask + right * neg_mask  # We have positive margin, reduce right hand side.
            last_right = mid * pos_mask + last_right * neg_mask  # Always sound, since the margin is positive.
            left = mid * neg_mask + left * pos_mask
        case2_d = BoundSin.d_func(last_right)
        case2_b = BoundSin.func(last_right) - case2_d * (last_right + cycles)
        d = case1_d * use_tangent_line + case2_d * (1. - use_tangent_line)
        b = case1_b * use_tangent_line + case2_b * (1. - use_tangent_line)
        # Return slope and bias.
        return [d, b]

    @staticmethod
    @torch.no_grad()
    def get_upper_left_bound(xl, steps=20):
        """Get a global upper bound given lower bound on x. Return slope and intercept."""
        dtype = xl.dtype
        # Constrain xl into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xl - 0.5 * torch.pi) / (2 * torch.pi)) * (2 * torch.pi)
        xl = xl - cycles
        use_tangent_line = (xl >= 2.0 * torch.pi).to(dtype)
        # Case 1: xl > pi, Lower tangent line is the only possible lower bound.
        case1_d = BoundSin.d_func(xl)
        case1_b = BoundSin.func(xl) - case1_d * (xl + cycles)
        # Case 2: Binary search needed. Testing from another tangent endpoint in [pi, 1.5*pi]. It must be in this region.
        left = (2.0 * torch.pi) * torch.ones_like(xl)
        # The right end guarantees the margin > 0 because it is basically a IBP lower bound (-1).
        right = (2.5 * torch.pi) * torch.ones_like(xl)
        last_right = right.clone()
        for _ in range(steps):
            mid = (left + right) / 2.
            margin = BoundSin.check_bound(mid, xl)
            pos_mask = (margin > 0).to(dtype)  # We want to margin < 0 but at small as possible.
            neg_mask = 1.0 - pos_mask
            right = mid * neg_mask + right * pos_mask  # We have positive margin, reduce right hand side.
            last_right = mid * neg_mask + last_right * pos_mask  # Always sound, since the margin is positive.
            left = mid * pos_mask + left * neg_mask
        case2_d = BoundSin.d_func(last_right)
        case2_b = BoundSin.func(last_right) - case2_d * (last_right + cycles)
        d = case1_d * use_tangent_line + case2_d * (1. - use_tangent_line)
        b = case1_b * use_tangent_line + case2_b * (1. - use_tangent_line)
        # Return slope and bias.
        return [d, b]

    @staticmethod
    @torch.no_grad()
    def get_lower_right_bound(xu, steps=20):
        """Get a global lower bound given upper bound on x. Return slope and intercept."""
        # Constrain xu into the -0.5 pi to 1.5 pi region.
        cycles = torch.floor((xu + 0.5 * torch.pi) / (2 * torch.pi)) * (2 * torch.pi)
        xu = xu - cycles
        d, _ = BoundSin.get_lower_left_bound(torch.pi - xu, steps)
        return [-d, BoundSin.func(xu) + d * (xu + cycles)]

    @staticmethod
    @torch.no_grad()
    def get_upper_right_bound(xu, steps=20):
        """Get a global upper bound given upper bound on x. Return slope and intercept."""
        # Constrain xu into the 0.5 pi to 2.5 pi region.
        cycles = torch.floor((xu - 0.5 * torch.pi) / (2 * torch.pi)) * (2 * torch.pi)
        xu = xu - cycles
        d, _ = BoundSin.get_upper_left_bound(3 * torch.pi - xu, steps)
        return [-d, BoundSin.func(xu) + d * (xu + cycles)]

    @staticmethod
    def interpoloate(x, lower_x, upper_x, lower_y, upper_y):
        x = torch.clamp(x, min=lower_x, max=upper_x)
        ratio = (x - lower_x) / (upper_x - lower_x).clamp(min=1e-8)
        return lower_y * (1. - ratio) + upper_y * ratio

    def get_bound_tb(self, tb, x):
        """Find lower or upper bounds from lookup table."""
        step = 2 * torch.pi / (BoundSin.n_table_entries - 1)
        # Move to 0 to 2 pi region.
        cycles = torch.floor(x / (2 * torch.pi)) * (2 * torch.pi)
        x = torch.clamp(x - cycles, min=0, max=2 * torch.pi)
        # Find the indice within the lookup table from 0 - 2pi.
        indices = x.div(step).long()
        # Intepolate the nearest d and b. This has better differentiability.
        # Another option is to always take the lower/upper side (better soundness).
        upper_indices = torch.clamp(indices + 1, max=BoundSin.n_table_entries-1)
        lower_x = self.all_table_x[indices]
        upper_x = self.all_table_x[upper_indices]
        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            self.alpha[ns].data[8:10, :] = torch.max(
                torch.min(self.alpha[ns][8:10, :], upper_x), lower_x)
            self.alpha[ns].data[10:12, :] = torch.max(
                torch.min(self.alpha[ns][10:12, :], upper_x), lower_x)
            self.alpha[ns].data[12:14, :] = torch.max(
                torch.min(self.alpha[ns][12:14, :], upper_x), lower_x)
            self.alpha[ns].data[14:16, :] = torch.max(
                torch.min(self.alpha[ns][14:16, :], upper_x), lower_x)

            lower_y_0 = self.alpha[ns][8:10, :]
            upper_y_0 = self.alpha[ns][10:12, :]
            lower_y_1 = self.alpha[ns][12:14, :]
            upper_y_1 = self.alpha[ns][14:16, :]
        else:
            lower_y_0 = tb[0][indices]
            upper_y_0 = tb[0][upper_indices]
            lower_y_1 = tb[1][indices]
            upper_y_1 = tb[1][upper_indices]
            if self.opt_stage == 'init':
                ns = self._start
                self.imp_both_upper_init_0[ns] = upper_y_0.detach()
                self.imp_both_lower_init_0[ns] = lower_y_0.detach()
                self.imp_both_upper_init_1[ns] = upper_y_1.detach()
                self.imp_both_lower_init_1[ns] = lower_y_1.detach()

        d = self.interpoloate(x, lower_x, upper_x, tb[0][indices], tb[0][upper_indices])
        b = self.interpoloate(x, lower_x, upper_x, tb[1][indices], tb[1][upper_indices])
        return d, b - d * cycles

    def forward(self, x):
        return torch.sin(x)

    def interval_propagate(self, *v):
        # Check if a point is in [l, u], considering the 2pi period
        def check_crossing(ll, uu, point):
            return ((((uu - point) / (2 * torch.pi)).floor()
                     - ((ll - point) / (2 * torch.pi)).floor()) > 0).to(h_Ls.dtype)
        h_L, h_U = v[0][0], v[0][1]
        h_Ls, h_Us = self.forward(h_L), self.forward(h_U)
        # If crossing pi/2, then max is fixed 1.0
        max_mask = check_crossing(h_L, h_U, self.ibp_max_point)
        # If crossing pi*3/2, then min is fixed -1.0
        min_mask = check_crossing(h_L, h_U, self.ibp_min_point)
        ub = torch.max(h_Ls, h_Us)
        ub = max_mask + (1 - max_mask) * ub
        lb = torch.min(h_Ls, h_Us)
        lb = - min_mask + (1 - min_mask) * lb
        return lb, ub

    def bound_relax_impl(self, lb, ub):
        dtype = lb.dtype

        ub = torch.max(ub, lb + 1e-8)

        # Case 1: Connect the two points as a line
        sub = self.func(ub)
        slb = self.func(lb)
        mid = (sub + slb) / 2.
        smid = self.func((ub + lb) / 2)
        gap = smid - mid

        min_preact = 1e-3
        mask_close = (ub - lb) < min_preact
        case1_line_slope = torch.where(mask_close, self.d_act_func(ub),
            (sub - slb) / (ub - lb).clamp(min=1e-10))
        case1_line_bias = slb - case1_line_slope * lb
        # Check if there are crossings between the line and the sin function.
        grad_crossings = self.get_intersection(lb, ub, case1_line_slope, theta=0.5 * torch.pi)
        # If there is no crossing, then we can connect the two points together as a lower/upper bound.
        use_line = grad_crossings == 1
        # Connected line is the upper bound.
        upper_use_line = torch.logical_and(gap <  0, use_line)
        # Connected line is the lower bound.
        lower_use_line = torch.logical_and(gap >= 0, use_line)
        # For the other bound, use the tangent line.
        case1_tangent_point = self.get_bounding_slope(lb, ub, case1_line_slope, theta=0.5 * torch.pi)
        case1_tangent_slope = case1_line_slope  # Use the same slope so far.
        stangent = self.func(case1_tangent_point)
        case1_tangent_bias = stangent - case1_tangent_slope * case1_tangent_point
        # Choose the lower/upper based on gap.
        case1_lower_slope = lower_use_line * case1_line_slope + upper_use_line * case1_tangent_slope
        case1_lower_bias = lower_use_line * case1_line_bias + upper_use_line * case1_tangent_bias
        case1_upper_slope = upper_use_line * case1_line_slope + lower_use_line * case1_tangent_slope
        case1_upper_bias = upper_use_line * case1_line_bias + lower_use_line * case1_tangent_bias

        # Case 2: we will try the global lower/upper bounds at lb and ub.
        # For the points and lb and ub, we can construct both lower and upper bounds.
        left_lower = self.get_bound_tb(BoundSin.xl_lower_tb, lb)  # slope, bias.
        left_upper = self.get_bound_tb(BoundSin.xl_upper_tb, lb)
        right_lower = self.get_bound_tb(BoundSin.xu_lower_tb, ub)
        right_upper = self.get_bound_tb(BoundSin.xu_upper_tb, ub)
        # Determine which lower bound is tighter.
        left_lower_error = sub - (left_lower[0] * ub + left_lower[1])
        right_lower_error = slb - (right_lower[0] * lb + right_lower[1])
        left_upper_error = (left_upper[0] * ub + left_upper[1]) - sub
        right_upper_error = (right_upper[0] * lb + right_upper[1]) - slb
        use_left_lower = (left_lower_error < right_lower_error).to(dtype)
        use_right_lower = 1. - use_left_lower
        use_left_upper = (left_upper_error < right_upper_error).to(dtype)
        use_right_upper = 1. - use_left_upper
        # Choose the slope and bias in this case.
        case_2_lower_slope = use_left_lower * left_lower[0] + use_right_lower * right_lower[0]
        case_2_lower_bias = use_left_lower * left_lower[1] + use_right_lower * right_lower[1]
        case_2_upper_slope = use_left_upper * left_upper[0] + use_right_upper * right_upper[0]
        case_2_upper_bias = use_left_upper * left_upper[1] + use_right_upper * right_upper[1]

        # Finally, choose between case 1 and case 2.
        use_line = use_line.to(dtype)
        not_use_line = 1. - use_line
        lower_slope = use_line * case1_lower_slope + not_use_line * case_2_lower_slope
        lower_bias = use_line * case1_lower_bias + not_use_line * case_2_lower_bias
        upper_slope = use_line * case1_upper_slope + not_use_line * case_2_upper_slope
        upper_bias = use_line * case1_upper_bias + not_use_line * case_2_upper_bias
        return lower_slope, lower_bias, upper_slope, upper_bias


class BoundCos(BoundSin):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.ibp_max_point = 0.0
        self.ibp_min_point = torch.pi

    def forward(self, x):
        return torch.cos(x)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        # Shift the input by half_pi, and shifting the linear bounds back.
        half_pi = 0.5 * torch.pi
        lb = x.lower + half_pi
        ub = x.upper + half_pi
        self.generate_inflections(lb, ub)
        self.branch_input_domain(lb, ub)
        self.bound_relax_impl_sigmoid(lb, ub, self.act_func, self.d_act_func)
        if self.opt_stage is None and self.sigmoid_like_mask.all():
            self.lb = self.lw * half_pi + self.lb
            self.ub = self.uw * half_pi + self.ub
            return
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_impl(lb, ub)
        self.lw = self.lw * self.sigmoid_like_mask + self.branch_mask * lower_slope
        self.lb = (self.sigmoid_like_mask * (self.lw * half_pi + self.lb)
                   + self.branch_mask * (lower_slope * half_pi + lower_bias))
        self.uw = self.uw * self.sigmoid_like_mask + self.branch_mask * upper_slope
        self.ub = (self.sigmoid_like_mask * (self.uw * half_pi + self.ub)
                   + self.branch_mask * (upper_slope * half_pi + upper_bias))


class BoundSec(BoundActivation):
    def forward(self, x):
        return 1. / torch.cos(x)

    def bound_relax(self, x, init=False):
        if init:
            self.init_linear_relaxation(x)

        assert x.lower.min() > -torch.pi / 2
        assert x.upper.max() < torch.pi / 2

        x_L = x.lower
        x_U = torch.max(x.upper, x_L + 1e-8)
        y_L = self.forward(x_L)
        y_U = self.forward(x_U)
        upper_k = (y_U - y_L) / (x_U - x_L)
        self.add_linear_relaxation(
            mask=None, type='upper', k=upper_k, x0=x_L, y0=y_L)

        mask_pos = x_L >= 0
        mask_neg = x_U <= 0
        mask_both = torch.logical_not(torch.logical_or(mask_pos, mask_neg))
        mid = (x_L + x_U) / 2
        y_mid = self.forward(mid)
        lower_k = y_mid * torch.tan(mid)
        self.add_linear_relaxation(mask=None, type='lower', k=lower_k,
                                   x0=mid, y0=y_mid)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        assert h_L.min() > -torch.pi / 2
        assert h_U.max() < torch.pi / 2
        y_L = self.forward(h_L)
        y_U = self.forward(h_U)
        lower = (h_U < 0) * (y_U - 1) + (h_L > 0) * (y_L - 1) + 1
        upper = torch.max(y_L, y_U)
        return lower, upper
