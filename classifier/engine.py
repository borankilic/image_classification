import torch.cuda
from typing import Dict, List

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_dataloader, epoch, optimizer, loss_fn, device=device, scheduler=None):
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # Send data and target to device
        data, target = data.to(device).type(torch.float32), target.to(device)
        # Forward pass
        output = model(data)
        # Calculate the loss
        loss = loss_fn(output, target)
        train_loss += loss
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backwards
        loss.backward()
        # Optimizer step
        optimizer.step()
        if scheduler:
            scheduler.step()
        # Calculate acc
        pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        train_acc += (pred_class == target).sum().item() / len(output)

        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.item()))
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc


def test(model, loss_fn, test_dataloader, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for data, target in test_dataloader:
            # Send data and target to device
            data, target = data.to(device).type(torch.float32), target.to(device)
            # Forward pass
            output = model(data)
            # Calculate loss
            loss = loss_fn(output, target)
            test_loss += loss.item()  # sum up batch loss

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_acc += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_acc, len(test_dataloader.dataset),
        100. * test_acc / len(test_dataloader.dataset)))
    test_acc /= len(test_dataloader.dataset)
    return test_loss, test_acc


def train_test_loop(model: torch.nn.Module
                    , epochs: int
                    , train_dataloader: torch.utils.data.DataLoader
                    , test_dataloader: torch.utils.data.DataLoader
                    , optimizer: torch.optim.Optimizer
                    , loss_fn: torch.nn.Module
                    , device=device
                    , scheduler=None) -> Dict[str, List]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train(model=model, epoch=epoch, optimizer=optimizer, loss_fn=loss_fn,
                                      train_dataloader=train_dataloader, device=device, scheduler=scheduler)
        test_loss, test_acc = test(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader, device=device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
