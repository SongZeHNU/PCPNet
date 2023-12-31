import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from datetime import datetime
from torchvision.utils import make_grid
from lib.PCPNetNet import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import cv2
from shutil import copyfile
from torch.optim import lr_scheduler
from skimage.filters.rank import entropy
from skimage.morphology import disk




def pfw_loss(pred, mask):
     
    a = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 1e-6
    b = torch.abs(F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask) + 1e-6
    c = torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask) + 1e-6
    d = torch.abs(F.avg_pool2d(mask, kernel_size=27, stride=1, padding=13) - mask) + 1e-6
    e = torch.abs(F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10) - mask) + 1e-6
 
    lam = 10.0
   
    fall = a**(0.5) + b**(0.5) + c**(0.5) + d**(0.5) + e**(0.5)
    a1 = (a**(0.5)  / fall) * a
    b1 = (b**(0.5) /fall ) * b
    c1 = (c**(0.5) /fall ) * c
    d1 = (d**(0.5) /fall ) * d
    e1 = (e**(0.5)  /fall) * e

    weight = torch.exp(a1+b1+c1+d1+e1) 
    dwbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    dwbce = (weight * dwbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    dwiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (dwbce + lam * dwiou).mean() 






def train(train_loader, model, optimizer, epoch, save_path, writer):
    #train function
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            

            preds = model(images)
            loss_final = pfw_loss(preds[0], gts) 
            loss_init = pfw_loss(preds[1], gts)
    

            loss = loss_final + loss_init

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                # res = preds[0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    #validation function
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _ = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate') #4.7e-5
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainValDataset/',    #./Dataset/TrainValDataset/   ./Dataset/DUTS-TR/
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/CHAMELEON/',  #./Dataset/TestDataset/CHAMELEON/   ./Dataset/DUTS-TEST/
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/PCPNet/',
                        help='the path to save model and log')

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    elif opt.gpu_id == '4':
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        print('USE GPU 4')
    elif opt.gpu_id == '5':
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        print('USE GPU 5')
    elif opt.gpu_id == '6':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 6')
    cudnn.benchmark = True

    # build the model
    model = Network(opt).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    #optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #record every run
    copyfile('./MyTrain_Val.py', save_path+'/MyTrain_Val.py')
    copyfile('./lib/PCPNet.py', save_path+'/PCPNet.py')

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batch_size,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batch_size, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch+1):
       
        train(train_loader, model, optimizer, epoch, save_path, writer)
        exp_lr_scheduler.step()
        val(val_loader, model, epoch, save_path, writer)
