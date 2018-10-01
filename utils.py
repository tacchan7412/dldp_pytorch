import torch
import math


def BatchClipByL2norm(t, upper_bound):
    batch_size = t.size(0)
    t2 = torch.reshape(t, (batch_size, -1))

    tensor = torch.tensor([])
    upper_bound_inv = tensor.new_full((batch_size,), 1.0/upper_bound).to(device)

    l2norm_inv = torch.rsqrt(torch.sum(t2*t2, 1) + 0.000001).to(device)

    scale = torch.min(upper_bound_inv, l2norm_inv) * upper_bound

    clipped_t = torch.mm(torch.diag(scale), t2)
    clipped_t = torch.reshape(clipped_t, t.size())
    return clipped_t

def AddGaussianNoise(t, sigma):
    if isinstance(t, torch.cuda.FloatTensor):
        noisy_t = t + torch.normal(mean=torch.zeros(t.size()), std=sigma).to(device)
    else:
        noisy_t = t + torch.normal(mean=torch.zeros(t.size()), std=sigma)
    return noisy_t

def GenerateBinomialTable(m):
    table = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        table[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            v = table[i - 1, j] + table[i - 1, j -1]
            assert not math.isnan(v) and not math.isinf(v)
            table[i, j] = v
    return torch.from_numpy(table)


