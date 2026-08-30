"""Unfused PyTorch AdamW equations, not torch.optim.AdamW's fused backend."""


def adamw_step(parameter, first, second, gradient, step, lr, beta1, beta2, epsilon, decay):
    first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    parameter.mul_(1.0 - lr * decay)
    denominator = (second / (1.0 - beta2**step)).sqrt().add_(epsilon)
    parameter.addcdiv_(first, denominator, value=-lr / (1.0 - beta1**step))
    return parameter, first, second
