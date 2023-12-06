import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.PCPNet import Network
from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=384, help='testing size')   #testsize
parser.add_argument('--pth_path', type=str, default='./snapshot/PCPNet/Net_epoch_100.pth') 

parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('USE GPU 0')
for _data_name in [ 'CAMO', 'COD10K', 'CHAMELEON','NC4K']:   #'CAMO', 'COD10K', 'CHAMELEON',
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)   #./Dataset/TestDataset
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network(opt)
  
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()


    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.trainsize) #trainsize

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        resf, _= model(image)
        res = resf 
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name,res*255)
