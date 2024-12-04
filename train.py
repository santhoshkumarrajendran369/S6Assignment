from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import Net
from torchsummary import summary

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item():.4f} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy

def main():
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Training settings
    batch_size = 128  # Increased batch size
    epochs = 15  # Reduced epochs
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    print("Downloading and loading MNIST dataset...")
    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0)),  # Random rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                    transform=test_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    print("Initializing model...")
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))

    # Initialize optimizer with cosine annealing scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best accuracy: {best_accuracy:.2f}%')

    print(f'\nBest Test accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()
