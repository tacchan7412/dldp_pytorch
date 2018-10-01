import utils
import collections

ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])

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
            saned_x = AddGaussianNoise(x, sigma * l2norm_bound)
        else:
            saned_x = x

        return saned_x


