import AlexNet
import AlexNet_InputData
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import time

data_path = "" # training dataset path
 

def AlexNet_Training():
    if torch.cuda.is_available():
        model = AlexNet.AlexNet().cuda()
    else:
        model = AlexNet.AlexNet()

    trainset = AlexNet_InputData.LoadDataset(data_path=data_path)
    # print(len(trainset))
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)

    lr = 1e-5
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(list(model.parameters())[:], lr=lr, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(10000):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        start_time = time.time()
        for trainData, trainLabel in train_loader:
            # trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
            trainData, trainLabel = trainData.cuda(), trainLabel.cuda()
            out = model(trainData)
            loss = loss_func(out, trainLabel)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == trainLabel).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #  if epoch % 100 == 0:
        print("speed time : {:.3f}".format((time.time() - start_time)))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            trainset)), train_acc / (len(trainset))))

        if (epoch + 1) % 10 == 0:
            sodir = './model/_iter_{}.pth'.format(epoch)
            print('[5] Model save {}'.format(sodir))
            torch.save(model.state_dict(), sodir)

        # adjust
        if (epoch + 1) % 100 == 0:
            lr = lr / 10
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)


if __name__ == "__main__":
    AlexNet_Training()
