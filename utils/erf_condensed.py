import os
import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter

def get_input_grad(model, samples, square=True):
    outputs = model(samples)
    out_size = outputs.size()
    if square:
        assert out_size[2] == out_size[3]
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def get_input_grad_avg(dataset, model: nn.Module, size=256, num_images=50, norms=lambda x:x):
    import tqdm
    from torchvision import datasets, transforms
    from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
    # transform = transforms.Compose([
    #     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    #     transforms.CenterCrop(size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ])
    # dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    data_loader_val = DataLoader(dataset, sampler=RandomSampler(dataset), pin_memory=True)

    meter = AverageMeter()
    model.cuda().eval()
    for _, (samples, _) in tqdm.tqdm(enumerate(data_loader_val)):
        if meter.count == num_images:
            break
        samples = samples.cuda(non_blocking=True).requires_grad_()
        As = samples['A']
        Bs = samples['B']
        label = samples['gt']
        name = samples['fn']
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)
        contribution_scores = get_input_grad(model, samples)
        if np.isnan(np.sum(contribution_scores)):
            print("got nan | ", end="")
            continue
        else:
            meter.update(contribution_scores)
    return norms(meter.avg)