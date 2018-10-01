import torch
import torch.optim as optim

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
