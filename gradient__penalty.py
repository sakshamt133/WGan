import torch
import utils


def gradient_penalty(real, fake, dis):

    inter = utils.epsilon * real + (1 - utils.epsilon) * fake
    out = dis(inter)

    grad = torch.autograd.grad(
        outputs=out,
        inputs=inter,
        grad_outputs=torch.ones_like(out),
        retain_graph=True,
        create_graph=True
    )[0]

    grad = grad.view(grad.shape[0], -1)
    gradient_norm = grad.norm(2, dim=1)
    gp = torch.mean((1 - gradient_norm) ** 2)
    return gp
