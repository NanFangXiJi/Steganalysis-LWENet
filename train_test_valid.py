import torch
import torch.nn.functional as F


def train(model, epoch, train_loader, batch_size, device, optimizer, scheduler, log_interval):
    total_loss = 0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(f"lr is now {lr_train}")

    model.train()
    for batch_idx, data in enumerate(train_loader):
        data, label = data['images'].to(device), data['labels'].to(device)

        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(train_loader.dataset) - batch_size * (len(train_loader) - 1)
            datas = data.view(last_batch_size * 2, 1, 256, 256)
            labels = label.view(last_batch_size * 2)
        else:
            datas = data.view(batch_size * 2, 1, 256, 256)
            labels = label.view(batch_size * 2)
        optimizer.zero_grad()
        output = model(datas)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()

            b_accu = b_correct / (labels.size(0))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), b_accu, loss.item()))
    print('train Epoch: {}\tavgLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))
    scheduler.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target,reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    accu = float(correct) / len(test_loader.dataset)
    return accu, test_loss


def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target,reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
    valid_loss, correct, len(valid_loader.dataset), 100. * correct / len(valid_loader.dataset)))
    accu = float(correct) / len(valid_loader.dataset)
    return accu, valid_loss


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

