import os.path

import torch.optim.optimizer
from torch import nn
from utils.data import VirtualBG_Dataset
from utils.transforms import Transforms
from torch.utils.data import DataLoader
from models.Unet_v1 import UNet
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from utils.utilities import Logger
import shutil

if __name__ == '__main__':
    pretrained_model = ''# r'C:\Users\meiyan\Desktop\VirtualBackground\log\models\UNet_sigmoid_20220612-002709\Model_UNet_sigmoid_Epoch-6_Iters-25200_loss-0.7344.pth'
    transform = Transforms(target_shape=(398,224))
    train_dataloader = DataLoader(VirtualBG_Dataset(transform=transform),
                                  batch_size=7, shuffle=True)
    val_dataloader = DataLoader(VirtualBG_Dataset('dataset/resized/val', transform=transform),
                                batch_size=4, shuffle=True)
    sigmoid_output = False  # Softmax的显存占用要高于Sigmoid
    model = UNet(3, sigmoid_output=sigmoid_output)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 18, 24, 30], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9, last_epoch=-1)

    if sigmoid_output:
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    TbWriter = SummaryWriter(f'log/{now}')
    shutil.copy(__file__, os.path.join(
        f'log/{now}',
        os.path.split(__file__)[-1]
    ))

    model_name = 'UNet_sigmoid'
    model_root = 'log/models/{}_{}'.format(model_name, now)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    logger = Logger(f'log/{model_name}_{now}.log')

    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
        optimizer.load_state_dict(torch.load(pretrained_model.replace('Model', 'Optim')))
        logger('Load Pretrained Model Successfully: {}'.format(pretrained_model))
    # 训练参数设置
    num_epochs = 40
    log_iters = 40
    save_iters = len(train_dataloader)
    eval_iters = 2000
    accumulation_iters = 40
    iter_counter = 0
    try:
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train_dataloader):
                iter_counter += 1
                if sigmoid_output == False:
                    y = 1 * (y >= 0.5)
                    y = y.long()  # https://www.csdn.net/tags/OtDaIg5sMTU1NzktYmxvZwO0O0OO0O0O.html 不需要自己转为one-hot编码，不过这样子得自己降低一个维度
                    # N_y, C_y, H_y, W_y = y.shape
                    y = torch.squeeze(y, dim=1)
                    # y_new = torch.zeros((N_y, 2, H_y, W_y), dtype=y.dtype)
                    # y_new[:, 0::2] = 1 * (y.detach() < 0.5)
                    # y_new[:, 1::2] = 1 * (y.detach() >= 0.5)
                    # y = y_new.long()  # https://blog.csdn.net/hhyys/article/details/110222728
                x = x.cuda()
                y = y.cuda()
                out = model(x)
                criterion_loss = criterion(out, y)  # 计算loss

                if sigmoid_output:
                    inner = out * y
                    union = out + y - inner
                    sum_axis = 2
                else:
                    inner = torch.squeeze(out[:, 1::2], 1) * y
                    union = torch.squeeze(out[:, 1::2], 1) + y - inner
                    sum_axis = 1
                iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(axis=sum_axis)
                iou = iou.mean()
                iou_loss = -torch.log(iou)  # https://blog.csdn.net/c2250645962/article/details/106053242

                loss = criterion_loss + iou_loss
                # backward
                if iter_counter and iter_counter % accumulation_iters == 0:
                    optimizer.zero_grad()  # 梯度归零
                    loss.backward()  # 方向传播
                    optimizer.step()  # 更新参数

                if iter_counter and iter_counter % eval_iters == 0:
                    with torch.no_grad():
                        loss_val = 0
                        mious = []
                        for i_val, (x_val, y_val) in enumerate(val_dataloader):
                            if sigmoid_output == False:
                                y_val = 1 * (y_val >= 0.5)
                                y_val = torch.squeeze(y_val, 1).long()
                            x_val = x_val.cuda()  # 造成val iou一直不变的原因，就说为什么val多大的batch size都可以
                            y_val = y_val.cuda()
                            out_val = model(x_val)
                            loss = criterion(out_val, y_val)
                            loss_val += loss.item()

                            # out_val = torch.argmax(out_val.detach())
                            if sigmoid_output:
                                out_val_segment = 1 * (out_val.detach().cpu().numpy() >= 0.5)
                                sum_axis = 2
                            else:
                                out_val_segment = 1 * (out_val[:, 1::2].detach().cpu().numpy() >= 0.5)
                                out_val_segment = np.squeeze(out_val_segment, 1)  # 20220612修改val iou全相等的bug
                                sum_axis = 1
                            y_val_segment = 1 * (y_val.detach().cpu().numpy() >= 0.5)
                            inner = out_val_segment * y_val_segment
                            union = 1 * ((out_val_segment + y_val_segment) >= 1)
                            iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(axis=sum_axis)
                            iou = iou.mean()
                            mious.append(iou)

                            logger('Epoch[{}/{}], Iter[{}/{}], Val Iter:[{}/{}], Val loss: {:.6f}, Val mIoU: {:.4f}'.format(
                                epoch + 1, num_epochs,
                                i + 1, len(train_dataloader),
                                i_val + 1, len(val_dataloader),
                                loss.data, iou))

                        loss_val /= i_val + 1
                        TbWriter.add_scalar('val/Criterion_loss', loss_val, iter_counter)

                        miou = np.mean(mious)
                        TbWriter.add_scalar('val/IoU_segment', miou, iter_counter)
                        TbWriter.add_scalar('val/IoU_segment_loss', -np.log(miou), iter_counter)
                        del mious
                        logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1,
                                                                                len(val_dataloader),
                                                                                loss.data))

                # if True:  # iter_counter % log_iters == 0:
                if iter_counter % log_iters == 0:
                    TbWriter.add_scalar('train/Criterion_loss', loss.item(), iter_counter)

                    img = x[0].cpu().numpy()[::-1]
                    prediction = out[0].cpu().detach().numpy()
                    TbWriter.add_image('train/image', img, iter_counter)
                    TbWriter.add_image('train/prediction', prediction, iter_counter)

                    if sigmoid_output:
                        out_segment = 1 * (out.detach().cpu().numpy() >= 0.5)
                        sum_axis = 2
                    else:
                        out_segment = 1 * (out[:, 1::2].detach().cpu().numpy() >= 0.5)
                        out_segment = np.squeeze(out_segment)
                        sum_axis = 1
                    y_segment = 1 * (y.detach().cpu().numpy() >= 0.5)
                    inner = out_segment * y_segment
                    union = 1 * ((out_segment + y_segment) >= 1)
                    iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(axis=sum_axis)
                    iou = iou.mean()
                    TbWriter.add_scalar('train/IoU_Segment', iou, iter_counter)
                    TbWriter.add_scalar('train/IoU_loss', iou_loss.item(), iter_counter)

                    TbWriter.add_scalar('train/learning_rate',
                                        optimizer.state_dict()['param_groups'][0]['lr'],
                                        iter_counter)
                    logger('=' * 20 + 'Train IoU Computation' + '=' * 20)
                    # logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}, iou: {:.2f}'.format(
                    #     epoch + 1, num_epochs,
                    #     i + 1, len(train_dataloader),
                    #     loss.data, iou.item()))
                    logger('=' * 20 + '=' * 21 + '=' * 20)

                if iter_counter and iter_counter % save_iters == 0:
                    model_path = 'Model_{}_Epoch-{}_Iters-{}_loss-{:.4f}.pth'.format(
                        model_name, epoch, iter_counter, loss.item())
                    optim_path = 'Optim_{}_Epoch-{}_Iters-{}_loss-{:.4f}.pth'.format(
                        model_name, epoch, iter_counter, loss.item())
                    model_path = os.path.join(model_root, model_path)
                    optim_path = os.path.join(model_root, optim_path)
                    torch.save(model.state_dict(), model_path)  # 还是不能偷懒直接存整个模型，可能是版本问题，可能暂时只有1.10版本可以
                    torch.save(optimizer.state_dict(), optim_path)
                    pass

                logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}, iou: {:.2f}'.format(
                    epoch + 1, num_epochs,
                    i + 1, len(train_dataloader),
                    loss.data, iou.item()))
            scheduler.step()
    except Exception as e:
        logger(str(e))
