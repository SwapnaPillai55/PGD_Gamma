# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:57:26 2024

@author: PRIA
"""

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

from glob import glob
import warnings
import time
import random
import numpy as np
import shutil
import torchvision.utils as vutils

from utils_image_VIRNet import batch_PSNR, batch_SSIM
#from utils1 import batch_PSNR, batch_SSIM

from tensorboardX import SummaryWriter
from math import ceil
from loss import loss_fn
from networks import weight_init_kaiming  #import VDN, weight_init_kaiming
from datasets import DenoisingDatasets_Gamma
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from options import set_opts

# Import evidential deep learning and ConvNext model
import evidential_deep_learning as edl
from Networks_Sid.models.convnext_isotropic_x2_f8_3c import convnext_isotropic_base


criterion2 = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

#from networks.DnCNN import DnCNN
#from networks.UNet import UNet
#from networks.VDN import VDN

# Fixing the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
seed = np.random.RandomState(112311)


from torch.distributions import Gamma




import torchvision.models as models

class PerceptualLossVGG16(nn.Module):
    def __init__(self, layers=[0, 5, 10, 17, 24]):
        super(PerceptualLossVGG16, self).__init__()
        
        # Load the pretrained VGG16 model
        vgg = models.vgg16(pretrained=True).features
        
        # Extract the layers corresponding to the perceptual loss
        self.layers = layers
        self.vgg_layers = nn.ModuleList([vgg[i] for i in layers])
        
        # Freeze the parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def preprocess_input(self, x):
        # Convert 1-channel image to 3-channel by repeating the channel 3 times
        #x = x.repeat(1, 3, 1, 1)
        
        # Normalization parameters for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        return x
    
    def forward(self, x, target):
        # Preprocess the input images
        x = self.preprocess_input(x)
        target = self.preprocess_input(target)
        
        # Extract features from different layers
        x_features = []
        target_features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            target = layer(target)
            x_features.append(x)
            target_features.append(target)
        
        # Compute the perceptual loss (L1 distance)
        loss = 0.0
        for xf, tf in zip(x_features, target_features):
            loss += torch.nn.functional.l1_loss(xf, tf)
        
        return loss




def tv_loss(x):
    """
    Compute the Total Variation Loss for a batch of images.
    Args:
    - x (torch.Tensor): A batch of images of shape (N, C, H, W)
    Returns:
    - torch.Tensor: The TV loss value
    """
    batch_size, channels, height, width = x.size()
    tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * channels * height * width)





class SimplifiedEvidentialGammaLoss(nn.Module):
    def __init__(self, regularization_coefficient):
        super(SimplifiedEvidentialGammaLoss, self).__init__()
        self.lambda_reg = regularization_coefficient

    def forward(self, alpha_pred, beta_pred, y_true):
        # Ensure the predicted parameters are positive and split y_pred into two parts
        #alpha_pred, beta_pred = torch.split(y_pred, y_pred.size(1) // 2, dim=1)
        
        # Apply softplus to ensure positivity
        alpha_pred = F.softplus(alpha_pred) + 1e-6
        beta_pred = F.softplus(beta_pred) + 1e-6
        y_true = y_true + 1e-6  # Ensure y_true is positive
        
        
        

        
        

        # Compute the NLL loss for the Gamma distribution using torch.lgamma for differentiable computation
        nll_loss = - (
            alpha_pred * torch.log(beta_pred)
            - torch.lgamma(alpha_pred)
            + (alpha_pred - 1) * torch.log(y_true)
            - beta_pred * y_true
        )

        # Compute the mean for regularization
        mu_pred = alpha_pred / beta_pred
        
        #perceptual_loss_fn = PerceptualLossVGG16().cuda()
        
       # perc_loss_value = perceptual_loss_fn(mu_pred, y_true)

        # Regularization term
        #reg_loss = torch.abs(y_true - mu_pred) * (2 * alpha_pred + beta_pred) # not applied 





        # Combine the losses
        #total_loss = nll_loss + (1e-3) * tv_loss_value#+ self.lambda_reg * reg_loss + (1e-1) * tv_loss_value
        #total_loss = nll_loss + (1e-3) * tv_loss_value#+ self.lambda_reg * reg_loss + (1e-1) * tv_loss_value
        total_loss = nll_loss  # + perc_loss_value#+ (1e-7) * tv_loss_value
        #total_loss = nll_loss * self.lambda_reg * reg_loss # + (1e-3) * tv_loss_value
        return torch.mean(total_loss)
    
    
# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
    return edl.losses.continuous.EvidentialRegression(true, pred, coeff=1e-2)

# Filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

# Set options and environment variables
args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_C = 3
_lr_min = 1e-6

#_modes = ['train', 'test_cbsd681', 'test_cbsd682', 'test_cbsd683']
_modes = ['train', 'test_bsd50']


def train_model(net, datasets, optimizer, lr_scheduler, criterion):
    clip_grad_next = args.clip_grad_D
    clip_grad_D = args.clip_grad_D
    clip_grad_S = args.clip_grad_S
    
    #batch_size = {'train': args.batch_size, 'test_cbsd681': 1, 'test_cbsd682': 1, 'test_cbsd683': 1}
    #batch_size = {'train': args.batch_size, 'test_bsd501': 1, 'test_bsd502': 1, 'test_bsd503': 1}
    batch_size = {'train': args.batch_size, 'test_bsd50': 1}

    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                                      shuffle=True, num_workers=args.num_workers, pin_memory=True) for phase in datasets.keys()}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    writer = SummaryWriter(args.log_dir)
    if args.resume:
        step = args.step
        step_img = args.step_img
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
        
    param_next = [x for name, x in net.named_parameters()]
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(args.epoch_start, args.epochs):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_next = 0
        tic = time.time()
        # Training stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        if lr < _lr_min:
            sys.exit('Reached the minimal learning rate')
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]

            optimizer.zero_grad()
            
            # Apply multiplicative noise
            # Apply multiplicative noise
            L = random.randint(10, 30) 
            #L = random.randint(10, 30) 
            
            noise = np.random.gamma(shape=L, scale=1.0/L, size=im_gt.shape)  # for num looks = 1
            im_noisy = ((im_gt.cpu().numpy() ** 1) * noise).astype(np.float32)
            im_noisy = torch.from_numpy(im_noisy)
            
            # Ensure im_noisy is on the same device as im_gt
            im_noisy = im_noisy.to(im_gt.device)
            #------------------------------------------------------------------------------------------------
            
            alpha_pred, beta_pred = net(im_noisy)

            
            # Apply softplus to ensure positivity
            alpha_pred = F.softplus(alpha_pred) + 1e-6
            beta_pred = F.softplus(beta_pred) + 1e-6
            im_gt = im_gt + 1e-6  # Ensure y_true is positive
        
        
            im_denoise = alpha_pred/beta_pred

            #print('im_den', im_gt)
                    
            #im_denoise = torch.clamp(im_denoise, 0.0, 1.0)

            
            #loss = EvidentialRegressionLoss(im_gt, outs)
            criterion = SimplifiedEvidentialGammaLoss(regularization_coefficient=0.01)
            #criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss =criterion(alpha_pred, beta_pred, im_gt)
 #
            #loss =criterion(im_denoise, im_gt)
            loss.backward()
            
            total_norm_next = nn.utils.clip_grad_norm_(param_next, clip_grad_next)
            grad_norm_next = (grad_norm_next * (ii / (ii + 1)) + total_norm_next / (ii + 1))
            optimizer.step()
            


            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            #im_denoise = torch.sqrt(mu+1e-10) #---------------------------------------------------------------------------------------
            #im_denoise = torch.clamp(im_denoise, 0.0, 1.0)

# =============================================================================
#             print('im_gt: ',im_gt)
#             print('im_denoise: ',im_denoise)
# =============================================================================

            mse = F.mse_loss(im_denoise, im_gt)
            mse_per_epoch[phase] += mse.item()
            
            if (ii + 1) % args.print_freq == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, mse={:.2e}, lr={:.1e}'
                print(log_str.format(epoch + 1, args.epochs, phase, ii + 1, num_iter_epoch[phase], mse, lr))
                
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse.item(), step)
                step += 1

        mse_per_epoch[phase] /= (ii + 1)
        log_str = '{:s}: Loss={:+.2e}, mse={:.3e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], mse_per_epoch[phase]))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        print('-' * 150)
        
        # Test stage
        net.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}
        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                im_noisy, im_gt = [x.cuda() for x in data]
                with torch.no_grad():
                    
                    L = 20#random.randint(4, 6)    
                    
                    noise = np.random.gamma(shape=L, scale=1.0/L, size=im_gt.shape)  # for num looks = 1
                    im_noisy = ((im_gt.cpu().numpy() ** 1) * noise).astype(np.float32)
                    im_noisy = torch.from_numpy(im_noisy)
                    im_noisy = im_noisy.to(im_gt.device)

                    alpha_pred, beta_pred = net(im_noisy)
                    #alpha_pred, beta_pred = torch.split(y_pred, y_pred.size(1) // 2, dim=1)
                    
                    alpha_pred = F.softplus(alpha_pred) + 1e-6
                    beta_pred = F.softplus(beta_pred) + 1e-6
                    im_gt = im_gt + 1e-6  # Ensure y_true is positive
                    
                    
                    im_denoise = alpha_pred / beta_pred
                    
                
                
                                    
                im_denoise = torch.clamp(im_denoise, 0.0, 1.0)
                im_gt = torch.clamp(im_gt, 0.0, 1.0)
                
                #------------------------------------------------
                
# =============================================================================
#                 im_gt_avg = torch.mean(im_gt, dim=1, keepdim=True)
#                 im_denoise_avg = torch.mean(im_denoise, dim=1, keepdim=True)
# =============================================================================

                #-------------------------------------------------
                
                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter

                if (ii + 1) % 20 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch + 1, args.epochs, phase, ii + 1, num_iter_epoch[phase], psnr_iter, ssim_iter))

            psnr_per_epoch[phase] /= (ii + 1)
            ssim_per_epoch[phase] /= (ii + 1)
            log_str = '{:s}: PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, psnr_per_epoch[phase], ssim_per_epoch[phase]))
        
        # Save the model only if PSNR and SSIM improved
        if psnr_per_epoch['test_bsd50'] > best_psnr and ssim_per_epoch['test_bsd50'] > best_ssim:
            print('\n ===== found better metrics =====')
            best_psnr = psnr_per_epoch['test_bsd50']
            best_ssim = ssim_per_epoch['test_bsd50']
            
            save_path_model = os.path.join(args.model_dir, f'model_{epoch + 1}')
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
                'step_img': {x: step_img[x] for x in _modes},
                'grad_norm_next': clip_grad_next,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
            save_path_model_state = os.path.join(args.model_dir, f'model_state_{epoch + 1}')
            torch.save(net.state_dict(), save_path_model_state)
            print(f"Model saved at epoch {epoch + 1} with PSNR: {psnr_per_epoch['test_bsd50']}, SSIM: {ssim_per_epoch['test_bsd50']}")
        
        writer.add_scalars('PSNR_epoch_test', psnr_per_epoch, epoch)
        writer.add_scalars('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print(f'This epoch took time {toc - tic:.2f}')
    
    writer.close()
    print('Reached the maximum epochs! Training complete.')

def main():
    # Build the model
    net = convnext_isotropic_base()
    #net = DnCNN()
    
    # Move the model to GPU
    net = nn.DataParallel(net).cuda()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    args.milestones = [10,  25,  35,  45] #[10, 20, 25, 30, 35, 40, 45, 50]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'=> Loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            args.step = checkpoint['step']
            args.step_img = checkpoint['step_img']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            
            #args.clip_grad_D = checkpoint['grad_norm_D']
            #args.clip_grad_S = checkpoint['grad_norm_S']
            print(f'=> Loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')
        else:
            sys.exit('Please provide the correct model path!')
    else:
        net = weight_init_kaiming(net)
        args.epoch_start = 0
        if os.path.isdir(args.log_dir):
            pass
        else:
            os.makedirs(args.log_dir)
        if os.path.isdir(args.model_dir):
            pass
        else:
            os.makedirs(args.model_dir)

    # Print the arg parameters
    for arg in vars(args):
        print(f'{arg:<15s}: {str(getattr(args, arg))}')

    # Prepare training data
    simulate_dir = Path(args.simulate_dir)
    train_im_list = list(simulate_dir.glob('*.jpg')) + list(simulate_dir.glob('*.png')) + list(simulate_dir.glob('*.bmp'))
    train_im_list = sorted([str(x) for x in train_im_list])
    # Prepare testing data
    
    
    test_case1_bsd50_h5 = Path('test_data').joinpath('noise_niid', 'BSD50_niid_case1.hdf5')

    #test_case2_h5 = Path('test_data').joinpath('noise_niid', 'BSD50_niid_case2.hdf5')
    #test_case3_h5 = Path('test_data').joinpath('noise_niid', 'BSD50_niid_case3.hdf5')
    test_im_list = (Path('test_data') / 'BSD50').glob('*.jpg')#'png')
    test_im_list = sorted([str(x) for x in test_im_list])
    
    
    
    datasets = {
        'train': DenoisingDatasets_Gamma.SimulateTrain(train_im_list, 5000 * args.batch_size, args.patch_size, radius=args.radius, noise_estimate=True),
        'test_bsd50': DenoisingDatasets_Gamma.SimulateTest(test_im_list, test_case1_bsd50_h5),
        
        #'test_bsd502': DenoisingDatasets_Gamma.SimulateTest(test_im_list, test_case2_h5),
        #'test_bsd503': DenoisingDatasets_Gamma.SimulateTest(test_im_list, test_case3_h5)
    }
    

    
    # Train model
    print(f'\nBegin training with GPU: {str(args.gpu_id)}')
    train_model(net, datasets, optimizer, scheduler, EvidentialRegressionLoss)

if __name__ == '__main__':
    main()
