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
    transform = Transforms()
    train_dataloader = DataLoader(VirtualBG_Dataset(transform=transform),
                                  batch_size=6, shuffle=True)
    val_dataloader = DataLoader(VirtualBG_Dataset('dataset/resized/val', transform=transform),
                                batch_size=64, shuffle=True)

    model = UNet(3)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 10, 20], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9, last_epoch=-1)
    criterion = nn.MSELoss().cuda()

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

    # 训练参数设置
    num_epochs = 40
    log_iters = 20
    save_iters = len(train_dataloader)
    eval_iters = 1500
    iter_counter = 0

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_dataloader):
            iter_counter += 1
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            mse_loss = criterion(out, y)  # 计算loss

            inner = out * y
            union = out + y - inner
            iou = inner.sum(axis=2).sum(axis=2) / union.sum(axis=2).sum(axis=2)
            iou = iou.mean()
            iou_loss = -torch.log(iou)  # https://blog.csdn.net/c2250645962/article/details/106053242

            loss = mse_loss+iou_loss
            # backward
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 方向传播
            optimizer.step()  # 更新参数

            if iter_counter and iter_counter % eval_iters == 0:
                with torch.no_grad():
                    loss_val = 0
                    mious = []
                    for i_val, (x_val, y_val) in enumerate(val_dataloader):
                        x_val = x.cuda()
                        y_val = y.cuda()
                        out_val = model(x_val)
                        loss = criterion(out_val, y_val)
                        loss_val += loss.item()

                        out_val_segment = 1 * (out_val.detach().cpu().numpy() >= 0.5)
                        y_val_segment = 1 * (y_val.detach().cpu().numpy() >= 0.5)
                        inner = out_val_segment * y_val_segment
                        union = 1 * ((out_val_segment + y_val_segment) >= 1)
                        iou = inner.sum(axis=2).sum(axis=2) / union.sum(axis=2).sum(axis=2)
                        iou = iou.mean()
                        mious.append(iou)

                        logger('Epoch[{}/{}], Iter[{}/{}], Val Iter:[{}/{}], Val loss: {:.6f}, Val mIoU: {:.2f}'.format(
                            epoch + 1, num_epochs,
                            i + 1, len(train_dataloader),
                            i_val + 1, len(val_dataloader),
                            loss.data, iou))

                    loss_val /= i_val + 1
                    TbWriter.add_scalar('val/MSEloss', loss_val, iter_counter)

                    miou = np.mean(mious)
                    TbWriter.add_scalar('val/IoU_segment', miou, iter_counter)
                    TbWriter.add_scalar('val/IoU_segment_loss', -np.log(miou),iter_counter)
                    del mious
                    logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1,
                                                                            len(val_dataloader),
                                                                            loss.data))

            # if True:  # iter_counter % log_iters == 0:
            if iter_counter % log_iters == 0:
                TbWriter.add_scalar('train/MSEloss', loss.item(), iter_counter)

                img = x[0].cpu().numpy()[::-1]
                prediction = out[0].cpu().detach().numpy()
                TbWriter.add_image('train/image', img, iter_counter)
                TbWriter.add_image('train/prediction', prediction, iter_counter)

                out_segment = 1 * (out.detach().cpu().numpy() >= 0.5)
                y_segment = 1 * (y.detach().cpu().numpy() >= 0.5)
                inner = out_segment * y_segment
                union = 1 * ((out_segment + y_segment) >= 1)
                iou = inner.sum(axis=2).sum(axis=2) / union.sum(axis=2).sum(axis=2)
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
