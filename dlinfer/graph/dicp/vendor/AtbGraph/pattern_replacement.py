import functools
import torch
from dlinfer.graph.dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)

atb_pattern_matcher = PatternMatcherPass()

torch_patterns_cls_list_1 = []
register_torch_pattern_1 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_1
)

torch_patterns_cls_list_2 = []
register_torch_pattern_2 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_2
)

torch_patterns_cls_list_3 = []
register_torch_pattern_3 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_3
)


aten = torch.ops.aten
atb = torch.ops.atb
dlinfer = torch.ops.dlinfer


@register_torch_pattern_1
class TorchLinear(BackendPatternBase):
    @staticmethod
    def pattern(x_input, weight, viewed_input_shape, viewed_output_shape):
        trans_weight = torch.ops.aten.t.default(weight)
        viewed_input = torch.ops.aten.view.default(x_input, viewed_input_shape)
        mm_result = torch.ops.aten.mm.default(viewed_input, trans_weight)
        viewed_mm_result = torch.ops.aten.view.default(mm_result, viewed_output_shape)
        return viewed_mm_result

    @staticmethod
    def replacement(x_input, weight):
        return torch.ops.atb.linear.default(x_input, weight, None, False, True)