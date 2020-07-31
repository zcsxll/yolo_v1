import os
import numpy as np
import torch
import torchvision.transforms as transforms

from model import create_model
from dataset import create_dataset
from loss import YoloLoss

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

    model = create_model('resnet50')()
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids = [i for i in range(4)])

    learning_rate = 0.001
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        # print(key)
        params += [{'params':[value], 'lr':learning_rate}]

    # print(params[-2])
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # print(transforms)
    # print(type(transforms.ToTensor()))
    # print(torch.nn.Module)
    dataset = create_dataset('voc2007_2012')(transform=[transforms.ToTensor()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True, num_workers=24)

    dataset_test = create_dataset('voc2007test')(transform=[transforms.ToTensor()], train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=False, num_workers=24)

    yloss = YoloLoss(5, 0.5).cuda()

    best_test_loss = np.inf
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()

        # if epoch == 1:
        #     learning_rate = 0.0005
        # if epoch == 2:
        #     learning_rate = 0.00075
        # if epoch == 3:
        #     learning_rate = 0.001
        if epoch == 30:
            learning_rate=0.0001
        if epoch == 40:
            learning_rate=0.00001

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        total_loss = 0.
        for idx, (img, target) in enumerate(dataloader):
            img, target = img.cuda(), target.cuda()

            pred = model(img)
            loss = yloss(pred, target)
            total_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx + 1) % 20 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                    %(epoch + 1, num_epochs, idx + 1, len(dataloader), loss.cpu().detach().numpy(), total_loss / (idx + 1)))
                # vis.plot_train_val(loss_train=total_loss/(i+1))
            # print(pred.shape)
        torch.save(model.state_dict(), 'yolo_v1_epoch%d.pth' % (epoch + 1))

        validation_loss = 0.0
        model.eval()
        for idx, (img, target) in enumerate(dataloader_test):
            img, target = img.cuda(), target.cuda()

            pred = model(img)
            loss = yloss(pred, target)
            validation_loss += loss.cpu().detach().numpy()
        validation_loss /= len(dataloader_test)
        # vis.plot_train_val(loss_val=validation_loss)
        
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(model.state_dict(), 'yolo_v1_best.pth')