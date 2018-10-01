import torch
import utils


def PCA_mat(data, with_privacy, sanitizer, sigma, n_components=60):
    data = data.view(-1, data.size(1) * data.size(2) * data.size(3))
    normalized_data = torch.t(torch.t(data) / torch.norm(data, p=2, dim=1))
    covar = torch.mm(torch.t(normalized_data), normalized_data)
    num_examples = data.size(0)
    if with_privacy:
        saned_covar = sanitizer.sanitize(torch.reshape(covar, (1,-1)), 
                                         sigma=sigma, 
                                         option=utils.ClipOption(1.0, False), 
                                         num_examples=num_examples)
        saned_covar = torch.reshape(saned_covar, covar.size())
        saned_covar = 0.5 * (saned_covar + torch.t(saned_covar))
    else:
        saned_covar = covar
    eigvals, eigvecs = torch.symeig(saned_covar, eigenvectors=True)
    _, topk_indices = torch.topk(eigvals, n_components)
    topk_indices = torch.reshape(topk_indices, (n_components,))
    return torch.t(torch.index_select(torch.t(eigvecs), 0, topk_indices))
