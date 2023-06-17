"""
version 1.5
    在1.4的基础上，添加如下功能： 当指定了pretrained_model时，将会找到对应的log给tensorboard续写。(因此完善了Iters的接续)

version 1.6
    在最后的1.5的基础上，模型保存时添加val loss的信息，并且在模型保存前都进行一次验证集的metrics计算。以应对过拟合时的模型选择。

version 1.7
    此版本在1.6的基础上，修改了transform，加入了随机旋转的数据增强模式。
    修复了val/prediction存储在tensorboard中时依然是前景为黑色的问题，这是因为prediction变量是对的，但是却add_image(xxx, val_out...[0],...)

version 1.8
    在1.7的基础上，给Unet模型添加了BN层可修改的接口。并在这次的训练中实验取消BN层的训练效果。
    并202206250309的训练结束后

version 1.9
    修正Unet中残差path中norm替换的bug, 同时修正模型保存中IoULoss和IoU保存交替出错的bug.

version 2.0
    这个版本主要是考虑更深的UNet能否解决之前的mIoU不够高和眼睛误判率高的问题，对应UNet_deep模型。

version 2.1
    这个版本将UNet_deep中的bottleneck的BN层也修改为IN层了。同时在这个训练代码中添加summary.
"""

import os.path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # https://blog.csdn.net/weixin_55593481/article/details/123617710
import torch.optim.optimizer
from torch import nn
from utils.data import VirtualBG_Dataset
from utils.transforms import Transforms
from torch.utils.data import DataLoader
from models.UNet_deep import UNet
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from utils.utilities import Logger
import shutil
from torchsummary import summary

if __name__ == '__main__':
    pretrained_model = None  # r'C:\Users\meiyan\Desktop\VirtualBackground\log\models\UNet_sigmoid_20220612-121150\Model_UNet_sigmoid_Epoch-20_Iters-64806_loss-0.9470.pth'
    sigmoid_output = True  # Softmax的显存占用要高于Sigmoid
    load_optim = True
    transform = Transforms(target_shape=(640, 360), convert2gray=0, rotate_range=[-45, 45])
    train_dataloader = DataLoader(VirtualBG_Dataset(transform=transform),
                                  batch_size=24, shuffle=True)
    val_dataloader = DataLoader(VirtualBG_Dataset('dataset/resized/val', transform=transform),
                                batch_size=1, shuffle=True)
    model = UNet(3, out_channels=8, sigmoid_output=sigmoid_output,
                 activation=nn.ReLU, residual=True, norm=nn.InstanceNorm2d).cuda()
    summary(model, (3, 640, 360))
    optimizer = torch.optim.Adam(model.parameters(), 0.005, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
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

    if pretrained_model:
        last_log = os.path.split(os.path.split(pretrained_model)[0])[-1].split('_')[-1]
        import glob

        last_log_file = glob.glob('log/' + last_log + '/*.0')[0]
        last_log_filename = os.path.split(last_log_file)[-1]
        shutil.copy(last_log_file, os.path.join('log', str(now), last_log_filename))

    model_name = 'UNet_sigmoid'
    model_root = 'log/models/{}_{}'.format(model_name, now)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    logger = Logger(f'log/{model_name}_{now}.log')

    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
        if load_optim:
            optimizer.load_state_dict(torch.load(pretrained_model.replace('Model', 'Optim')))
        logger('Load Pretrained Model Successfully: {}'.format(pretrained_model))
    # 训练参数设置
    num_epochs = 80
    log_iters = 40
    save_iters = len(train_dataloader)
    eval_iters = 30  # 400
    accumulation_iters = 1
    if pretrained_model:
        iter_counter = int(pretrained_model.split('Iters-')[-1].split('_')[0])
    else:
        iter_counter = 0
    try:
        model.train()
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train_dataloader):
                # print('model.training: ',model.training)
                # print('INFO INSPECTING, x.max,', x.cpu().numpy().max())
                # logger('down_block1.bn0.running_mean:{}'.format(model.down_block1.block[1].running_mean))
                # logger('down_block1.bn0.running_var:{}'.format(model.down_block1.block[1].running_var))
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
                # loss = iou_loss
                # backward
                if iter_counter and iter_counter % accumulation_iters == 0:
                    optimizer.zero_grad()  # 梯度归零
                    loss.backward()  # 方向传播
                    optimizer.step()  # 更新参数

                if iter_counter and iter_counter % eval_iters == 0:
                    model.eval()
                    with torch.no_grad():
                        loss_val = 0
                        mious = []
                        for i_val, (x_val, y_val) in enumerate(val_dataloader):
                            print(model.training)
                            # logger('down_block1.bn0.running_mean:{}'.format(model.down_block1.block[1].running_mean))
                            # logger('down_block1.bn0.running_var:{}'.format(model.down_block1.block[1].running_var))
                            # print('INFO INSPECTING, x.max,', x_val.cpu().numpy().max())
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
                            iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(
                                axis=sum_axis)
                            iou = iou.mean()
                            mious.append(iou)

                            logger(
                                'Epoch[{}/{}], Iter[{}/{}], Val Iter:[{}/{}], Val loss: {:.6f}, Val mIoU: {:.4f}'.format(
                                    epoch + 1, num_epochs,
                                    i + 1, len(train_dataloader),
                                    i_val + 1, len(val_dataloader),
                                    loss.data, iou))

                        loss_val /= i_val + 1
                        TbWriter.add_scalar('val/Criterion_loss', loss_val, iter_counter)

                        miou = np.mean(mious)
                        TbWriter.add_scalar('val/IoU_segment', miou, iter_counter)
                        TbWriter.add_scalar('val/IoU_segment_loss', -np.log(miou), iter_counter)
                        TbWriter.add_image('val/image', x_val.detach().cpu().numpy()[0][::-1], iter_counter)
                        prediction = out_val.detach().cpu().numpy()[0]
                        label_val = y_val.detach().cpu().numpy()[0]
                        if not sigmoid_output:
                            prediction = prediction[1][None, :, :]
                            label_val = label_val[None, :, :]
                        TbWriter.add_image('val/prediction', prediction, iter_counter)
                        TbWriter.add_image('val/label', label_val, iter_counter)
                        del mious
                        logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1,
                                                                                len(val_dataloader),
                                                                                loss.data))
                        model.train()

                # if True:  # iter_counter % log_iters == 0:
                if iter_counter % log_iters == 0:
                    TbWriter.add_scalar('train/Criterion_loss', loss.item(), iter_counter)

                    img = x[0].cpu().numpy()[::-1]
                    prediction = out[0].cpu().detach().numpy()
                    label = y.detach().cpu().numpy()[0]
                    if not sigmoid_output:
                        prediction = prediction[1][None, :, :]
                        label = label[None, :, :]
                    TbWriter.add_image('train/image', img, iter_counter)
                    TbWriter.add_image('train/prediction', prediction, iter_counter)
                    TbWriter.add_image('train/label', label, iter_counter)

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

                    model_params = model.state_dict()
                    for param_key in model_params:
                        TbWriter.add_histogram('net/{}'.format(param_key), model_params[param_key], iter_counter)

                    logger('=' * 20 + 'Train IoU Computation' + '=' * 20)
                    # logger('Epoch[{}/{}], Iter[{}/{}], loss: {:.6f}, iou: {:.2f}'.format(
                    #     epoch + 1, num_epochs,
                    #     i + 1, len(train_dataloader),
                    #     loss.data, iou.item()))
                    logger('=' * 20 + '=' * 21 + '=' * 20)

                if iter_counter and iter_counter % save_iters == 0:
                    # model.eval()
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
                            iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(
                                axis=sum_axis)
                            iou = iou.mean()
                            mious.append(iou)

                        loss_val /= i_val + 1
                        TbWriter.add_scalar('val/Criterion_loss', loss_val, iter_counter)

                        miou = np.mean(mious)
                        TbWriter.add_scalar('val/IoU_segment', miou, iter_counter)
                        TbWriter.add_scalar('val/IoU_segment_loss', -np.log(miou), iter_counter)
                    model.train()

                    model_path = 'Model_{}_Epoch-{}_Iters-{}_loss-{:.4f}_criterionLoss-{:.4f}_IoULoss-{:.4f}_IoU-{:.4f}.pth'.format(
                        model_name, epoch, iter_counter, loss.item(), loss_val, -np.log(miou), miou)
                    optim_path = 'Optim_{}_Epoch-{}_Iters-{}_loss-{:.4f}_criterionLoss-{:.4f}_IoULoss-{:.4f}_IoU-{:.4f}.pth'.format(
                        model_name, epoch, iter_counter, loss.item(), loss_val, -np.log(miou), miou)
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
