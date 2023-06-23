import torch
import math


# defining he_normalizer and uniformer of tensorflow
def calculate_correct_fan(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    return fan_in


def he_normalizer(model, mode="fan_in"):
    tensor = model.weight
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor

    fan = calculate_correct_fan(tensor)
    std = math.sqrt(2.0 / fan)
    with torch.no_grad():
        return tensor.normal_(0, std)