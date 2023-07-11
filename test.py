import glob
from network_style import *
import torchvision.utils as utils
import numpy as np
from measure import compute_measure
from model import *
from prep import printProgressBar
from loader import get_loader
import argparse
import torch
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see  issue
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--load_mode', type=int, default=0)
parser.add_argument('--data_path', type=str, default='/home/sutanu/CT_d/CT')
parser.add_argument('--saved_path', type=str,
                    default='/home1/sutanu/CT_big/test_1mm_ab/')
parser.add_argument('--save_path', type=str,
                    default='/home/sutanu/CT_d/RealCT/save/')
parser.add_argument('--test_patient', type=str, default='L506')
parser.add_argument('--result_fig', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=1024.0)
parser.add_argument('--trunc_min', type=float, default=-140.0)
parser.add_argument('--trunc_max', type=float, default=260.0)

parser.add_argument('--transform', type=bool, default=False)

# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=4)


parser.add_argument('--num_workers', type=int, default=7)


args = parser.parse_args()

def save_fig(x, y, pred, fig_name, original_result, pred_result):
    x, y, pred = np.squeeze(x), np.squeeze(y), np.squeeze(pred)
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray,
                 vmin=args.trunc_min, vmax=args.trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(args.save_path,
                           'result_{}.png'.format(fig_name)))
    plt.close()
    
    
# load


device = torch.device("cuda:0")
#

model_test = AuxDenoiserTest(image_size)

# Load the saved model weights
checkpoint = torch.load(args.save_path +  'model_weights.pth')

# Load the weights into the corresponding model components
model_test.encoder.load_state_dict(checkpoint['encoder_state_dict'])
model_test.decoder.load_state_dict(checkpoint['decoder_state_dict'])


# compute PSNR, SSIM, RMSE
ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

k = 0


file_ldct = sorted([f for f in glob.glob(args.saved_path + "/*_input.npy")])
file_ndct = sorted([f for f in glob.glob(args.saved_path + "/*_target.npy")])
with torch.no_grad():
    for idx in range(len(file_ndct)):

        netG.eval()

        x = torch.tensor(np.load(file_ldct[idx]))
        y = torch.tensor(np.load(file_ndct[idx]))
        shape_ = x.shape[-1]

        x = x.unsqueeze(0).unsqueeze(0).float().cuda()
        y = y.unsqueeze(0).unsqueeze(0).float().cuda()

      #
        with torch.no_grad():
            pred = netG(x)

        x = trunc(denormalize_(x.cpu().detach(), args.norm_range_max, args.norm_range_min), args.trunc_max, args.trunc_min)
        y = trunc(denormalize_(y.cpu().detach(), args.norm_range_max, args.norm_range_min), args.trunc_max, args.trunc_min)
        pred = trunc(denormalize_(pred.cpu().detach(), args.norm_range_max, args.norm_range_min), args.trunc_max, args.trunc_min)


        data_range = args.trunc_max - args.trunc_min
        original_result, pred_result = compute_measure(x, y, pred, data_range)

        ori_psnr_avg += original_result[0]
        ori_ssim_avg += original_result[1]
        ori_rmse_avg += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]

        # save result figure
        if args.result_fig:
            save_fig(x, y, pred, idx, original_result, pred_result)
            pred = normalize_(pred, args.trunc_min, args.trunc_max)
            pred = torch.Tensor(pred)
            utils.save_image(pred, os.path.join(
                args.save_path, 'fig', 'Pred_{}.png'.format(idx)))

        printProgressBar(idx, len(file_ldct),
                         prefix="Compute measurements ..",
                         suffix='Complete', length=25)
    print('\n')
    print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(file_ndct),
                                                                                    ori_ssim_avg /
                                                                                    len(file_ndct),
                                                                                    ori_rmse_avg/len(file_ndct)))
    print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(file_ndct),
                                                                                          pred_ssim_avg /
                                                                                          len(
                                                                                              file_ndct),
                                                                                          pred_rmse_avg/len(file_ndct)))
