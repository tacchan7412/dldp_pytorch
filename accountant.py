import torch
import numpy as np
import collections
import math
import utils

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

class GaussianMomentsAccountant(object):
    def __init__(self, total_examples, moment_orders=32):
        self.total_examples = total_examples
        self.moment_orders = range(1, moment_orders+1)
        self.max_moment_order = max(self.moment_orders)
        self.log_moments = torch.zeros(self.max_moment_order, dtype=torch.float64)
        self.binomial_table = utils.GenerateBinomialTable(self.max_moment_order)
        
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
 
