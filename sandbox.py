import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PCA_mat(data, with_privacy, sanitizer, sigma, n_components=60):
    data = data.view(-1, data.size(1) * data.size(2) * data.size(3))
    normalized_data = torch.t(torch.t(data) / torch.norm(data, p=2, dim=1))
    covar = torch.mm(torch.t(normalized_data), normalized_data)
    num_examples = data.size(0)
    if with_privacy:
        saned_covar = sanitizer.sanitize(torch.reshape(covar, (1,-1)), 
                                         sigma=sigma, 
                                         option=ClipOption(1.0, False), 
                                         num_examples=num_examples)
        saned_covar = torch.reshape(saned_covar, covar.size())
        saned_covar = 0.5 * (saned_covar + torch.t(saned_covar))
    else:
        saned_covar = covar
    eigvals, eigvecs = torch.symeig(saned_covar, eigenvectors=True)
    _, topk_indices = torch.topk(eigvals, n_components)
    topk_indices = torch.reshape(topk_indices, (n_components,))
    return torch.t(torch.index_select(torch.t(eigvecs), 0, topk_indices))

class Net(nn.Module):
    def __init__(self, U):
        super(Net, self).__init__()
        self.U = U
        self.fc1 = nn.Linear(60, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.mm(x, self.U)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import collections

ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])

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
#     print(t)
    if isinstance(t, torch.cuda.FloatTensor):
        noisy_t = t + torch.normal(mean=torch.zeros(t.size()), std=sigma).to(device)
    else:
        noisy_t = t + torch.normal(mean=torch.zeros(t.size()), std=sigma)
#     print(noisy_t)
    return noisy_t

class AmortizedGaussianSanitizer(object):
    def __init__(self, accountant, default_option):
        self.accountant = accountant
        self.default_option = default_option
    
    def sanitize(self, x, sigma, option=ClipOption(None, None), num_examples=None, add_noise=True):
        l2norm_bound, clip = option
        if l2norm_bound is None:
            l2norm_bound, clip = self.default_option
        l2norm_bound_ = torch.tensor(l2norm_bound).to(device)
        if clip:
            x = BatchClipByL2norm(x, l2norm_bound_)
        if add_noise:
            self.accountant.accumulate_privacy_spending(sigma, num_examples)
            saned_x = AddGaussianNoise(x, 
                                       sigma * l2norm_bound)
        else:
            saned_x = x

        return saned_x

import collections
import math

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

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

class GaussianMomentsAccountant(object):
    def __init__(self, total_examples, moment_orders=32):
        self.total_examples = total_examples
        self.moment_orders = range(1, moment_orders+1)
        self.max_moment_order = max(self.moment_orders)
        self.log_moments = torch.zeros(self.max_moment_order, dtype=torch.float64)
        self.binomial_table = GenerateBinomialTable(self.max_moment_order)
        
    def accumulate_privacy_spending(self, sigma, num_examples):
        q = num_examples * 1.0 / self.total_examples
        for i in range(self.max_moment_order):
            moment = self.compute_log_moment(sigma, q, self.moment_orders[i])
            self.log_moments[i].add_(moment)
    
    def compute_log_moment(self, sigma, q, moment_order):
        binomial_table = self.binomial_table[moment_order:moment_order+1, :moment_order+1]
        qs = torch.exp(torch.tensor([i * 1.0 for i in range(moment_order+1)], 
                                    dtype=torch.float64) * torch.log(torch.tensor(q, dtype=torch.float64)))
        moments0 = self.differential_moments(sigma, 0.0, moment_order)
        term0 = torch.sum(binomial_table * qs * moments0)
        moments1 = self.differential_moments(sigma, 1.0, moment_order)
        term1 = torch.sum(binomial_table * qs * moments1)
        return torch.log(q * term0 + (1.0 - q) * term1)
    
    def differential_moments(self, sigma, s, t):
        binomial = self.binomial_table[:t+1, :t+1]
        signs = np.zeros((t + 1, t + 1), dtype=np.float64)
        for i in range(t+1):
            for j in range(t+1):
                signs[i, j] = 1.0 - 2 * ((i-j) % 2)
        exponents = torch.tensor([i * (i + 1.0 - 2.0 * s) / (2.0 * sigma * sigma) 
                                  for i in range(t+1)], dtype=torch.float64)
        x = torch.mul(binomial, torch.from_numpy(signs))
        y = torch.mul(x, torch.exp(exponents))
        z = torch.sum(y, 1)
        return z
    
    def get_privacy_spent(self, target_deltas):
        eps_deltas = []
        for delta in target_deltas:
            log_moments_with_order = zip(self.moment_orders, self.log_moments)
            eps_deltas.append(EpsDelta(self.compute_eps(log_moments_with_order, delta), delta))
        return eps_deltas
    
    def compute_eps(self, log_moments, delta):
        min_eps = float("inf")
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
        return min_eps
        
class DPSGD(optim.SGD):
    def __init__(self, sanitizer, sigma, batches_per_lot, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(DPSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.batches_per_lot = batches_per_lot  # assume 1
        self.grad_accum_dict = {}
        self.sanitizer = sanitizer
        self.sigma = sigma
        
    def compute_sanitized_gradients(self, loss, add_noise=True):
        px_grads = loss  # TODO: per_example_gradients.
                         # now assumes batch_size = 1
        sanitized_grads = self.sanitizer.sanitize(px_grads, self.sigma, add_noise=add_noise, num_examples=self.batches_per_lot * px_grads.size(0))
        
        return sanitized_grads

    def step(self, closure=None):
        '''
        override step method
        '''
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                """
                modified the line below
                old: d_p = p.grad.data
                """
                d_p = self.compute_sanitized_gradients(p.grad.data)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    if with_privacy:
        spent_eps_deltas = priv_accountant.get_privacy_spent(target_deltas=[target_delta])
        for spent_eps, spent_delta in spent_eps_deltas:
            print("spent privacy: eps %.4f delta %.5g\n" % (
              spent_eps, spent_delta))

batch_size = 1
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.0
log_interval = 1000

with_privacy = True
sigma = 4.0 #4.0
pca_sigma = 7.0 #7.0
target_delta = 1e-5
batches_per_lot = 1
default_gradient_l2norm_bound = 4.0 #4.0

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)

priv_accountant = GaussianMomentsAccountant(60000)
gaussian_sanitizer = AmortizedGaussianSanitizer(
        priv_accountant,
        [default_gradient_l2norm_bound / batch_size, True])

# pca init
pca_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=60000, shuffle=True)
all_data, _ = iter(pca_train_loader).next()
U = PCA_mat(all_data, with_privacy, gaussian_sanitizer, pca_sigma).to(device)

model = Net(U).to(device)
if with_privacy:
    optimizer = DPSGD(sanitizer=gaussian_sanitizer, sigma=sigma, batches_per_lot=batches_per_lot, params=model.parameters(), lr=lr, momentum=momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
