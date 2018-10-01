import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
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

if __name__ == '__main__':
    main()
