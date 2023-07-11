import argparse
from loader import get_loader
import torch
import time
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import vgg19
import torch.nn as nn
from model import *
import os
import numpy as np
import loader_single
from utils import *


torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--load_mode', type=int, default=0)
parser.add_argument('--save_path', type=str, default='/home/sutanu/CT_d/RealCT/save/')
parser.add_argument('--path_ldct', type=str, default='/home1/sutanu/CT_big/train_chest')
parser.add_argument('--path_real', type=str, default='/home1/sutanu/CT_big/train_real/')

parser.add_argument('--scale_data', type=bool, default=False)
parser.add_argument('--scale_factor', type = float, default = 1.3)

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=3)

parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--print_iters', type=int, default=20)
parser.add_argument('--decay_iters', type=int, default=6000)
parser.add_argument('--save_iters', type=int, default=1000)
parser.add_argument('--test_iters', type=int, default=1000)

parser.add_argument('--device', type=str)
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for adam')
parser.add_argument('--load_chkpt', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=7)

args = parser.parse_args()

  

syn_loader= get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.path_ldct,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1))

real_loader = loader_single.get_loader(mode=args.mode,
                             load_file = 'input',
                             load_mode=args.load_mode,
                             saved_path=args.path_real,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1))
if args.load_chkpt:
    print('Loading Chekpoint')
    whole_model = torch.load(args.save_path+ 'latest_ckpt.pth.tar')
    netG_state_dict,optG_state_dict = whole_model['netG_state_dict'], whole_model['optG_state_dict']
    netG = BaseDenoiser(args.patch_size)
    netG = netG.cuda()
    optG = torch.optim.Adam(netG.parameters())
    netG.load_state_dict(netG_state_dict)
    optG.load_state_dict(optG_state_dict)
    cur_epoch = whole_model['epoch']
    total_iters = whole_model['total_iters']
    lr = whole_model['lr']
    netG = torch.nn.DataParallel(netG, device_ids=[0, 1])
    print(cur_epoch)
else:
    print('Training model from scratch')
    netG = AuxDenoiser(args.patch_size)
    netG = netG.cuda()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr)
    cur_epoch = 0
    total_iters = 0
    lr=args.lr
 
    
train_losses = []
criterionMSE= nn.MSELoss()
criterionMSE = criterionMSE.cuda()
criterionBCE=nn.CrossEntropyLoss()
criterionL1= nn.L1Loss().cuda()

domain_labels = create_domain_labels(args.batch_size, args.patch_n)

start_time = time.time()
generator2 = iter(real_loader)
generator1 = iter(syn_loader)

max_iter = max(len(real_loader),len(syn_loader))

for epoch in range(cur_epoch, args.num_epochs):
    netG.train()
    # netD.train()
    print(epoch)
    for iter_ in range(max_iter):
        total_iters += 1

        try:
            # Samples the batch
            x_syn, y_syn = next(generator1)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator1 = iter(syn_loader)
            x_syn, y_syn = next(generator1)
        # add 1 channel
        
        try:
            # Samples the batch
            x_real = next(generator2)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator2 = iter(real_loader)
            x_real = next(generator2)   
        
        if args.scale_data == True:
                x_syn = scale_data(x_syn, args.scale_factor)
                y_syn = scale_data(y_syn, args.scale_factor)
        
        x_real = x_real.unsqueeze(0).float()
        x_syn = x_syn.unsqueeze(0).float()
        y_syn = y_syn.unsqueeze(0).float()


        if args.patch_size: # patch training
            x_real = x_real.view(-1, 1, args.patch_size, args.patch_size)
            y_syn = y_syn.view(-1, 1, args.patch_size, args.patch_size)
            x_syn = x_syn.view(-1, 1, args.patch_size, args.patch_size)


        x_real = x_real.cuda()
        y_syn = y_syn.cuda()
        x_syn = x_syn.cuda()

        
        inp_imgs=torch.cat((x_real,x_syn), dim=0)
        
        
        pred, logits , aux1, aux2 = netG(inp_imgs)
        
        pred_real, pred = torch.chunk(pred, 2, dim=0)
        
        pred_real_aux1 , pred_syn_aux1= torch.chunk(aux1, 2, dim=0)
        pred_real_aux2 , pred_syn_aux2= torch.chunk(aux2, 2, dim=0)
    
    
            
        optG.zero_grad()
        netG.zero_grad()


        aux_loss1 = criterionMSE(pred_real,pred_real_aux1) 
        aux_loss2 = criterionMSE(pred_real,pred_real_aux2)  
        aux_loss =  0.5 * (aux_loss1+aux_loss2)
        r_loss = criterionMSE(pred, y_syn)
        c_loss = criterionBCE(logits,domain_labels)
        loss =  r_loss +  c_loss +  aux_loss
        loss.backward()
            
        optG.step()
    
        if total_iters % 20 ==0:
            print("===> Epoch[{}]({}/{}): R_Loss: {:.8f} CLoss:  {:.10f} ALoss: {:.8f} ".format(
                epoch, iter_, max_iter, r_loss.item(), c_loss.item(), aux_loss.item()))
        
        # learning rate decay
        
        if total_iters % args.decay_iters == 0:
            lr = lr * 0.5
            for param_group in optG.param_groups:
                param_group['lr'] = lr
                
        # save model      



# Save the weights of the encoder and decoders
    saved_model = {
        'encoder_state_dict': netG.encoder.state_dict(),
        'decoder_state_dict': netG.decoder.state_dict()
    }

    torch.save(saved_model, '{}model_weights.pth'.format(args.save_path))
    
# Save the checkpoint for resuming the training. 
                  
    saved_model = {
                   'epoch': epoch ,
                   'netG_state_dict': netG.state_dict(),
                   'optG_state_dict': optG.state_dict(),
                   'lr': lr,
                   'total_iters': total_iters,
                   'epoch': epoch
                   }
    torch.save(saved_model, '{}Epoch_{}_checkpoint.pth.tar'.format(args.save_path, epoch))