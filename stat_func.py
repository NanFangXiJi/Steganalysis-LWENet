import torch
from torch.autograd import Variable
import torch.nn.functional as F


def sum(pred, target):
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])

    cover_num = len(target[target == 0])

    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2), cover_num, len(target) - cover_num


def test_stat(model, device, test_loader):
    model.eval()
    loss = 0
    correct = 0.
    N = 0  # 正确被分类为载体图像的数目
    P = 0  # 正确被分类为载密图像的数目
    C = 0  # 待测数据集中所有载体图像的个数
    S = 0  # 待测数据中所有的载密图像个数

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            a, b, c, d, e = sum(pred, target)
            N += a
            P += b
            correct += c
            C += d
            S += e
    loss /= len(test_loader.dataset)
    accu = float(correct) / len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
    loss, correct, len(test_loader.dataset), 100. * accu))
    FPR = (C - N) / C  # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
    Pmd = (S - P) / S  # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率
    print('虚警率(FPR): {}/{} ({:.6f}%)'.format(C - N, C, 100. * FPR))
    print('漏检率(FNR): {}/{} ({:.6f}%)'.format(S - P, S, 100. * Pmd))
    return accu, loss

