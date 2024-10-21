import torch
import torchvision
import argparse
import numpy as np
import logging
import time
import os
import random
import timm
import copy
from tqdm import tqdm
from sam import SAM
from torch import nn, optim
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from itertools import accumulate
from torch.utils.data import Subset
from models.wideresnet import *
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class reProgrammingNetwork(nn.Module):
    def __init__(self,args, input_size=224, patch_H_size=192, patch_W_size=192, channel_out=3, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.channel_out = channel_out
        self.input_size = input_size
        if args.model_name == 'wideresnet':
            self.pre_model = torchvision.models.wide_resnet50_2(pretrained=True)
        elif args.model_name == 'resnet50':
            self.pre_model = torchvision.models.resnet50(pretrained=True)
        elif args.model_name == 'resnet152':
            self.pre_model = torchvision.models.resnet152(pretrained=True)
        elif args.model_name == 'swin':
            self.pre_model = torchvision.models.swin_v2_s(pretrained=True)
        elif args.model_name == 'vit':
            self.pre_model = torchvision.models.vit_b_32(pretrained=True)
        elif args.model_name == 'vit_21k':
            self.pre_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.model_name == 'swin_22k':
            self.pre_model = SwinForImageClassification.from_pretrained("microsoft1/swin-base-patch4-window7-224-in22k/")
        elif args.model_name == 'swinv2_22k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        elif args.model_name == 'swinv2_22k_ft_1k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft")
        elif args.model_name == 'swinv2_large_22k':
            self.pre_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
        elif args.model_name == 'convnextv2_ft_in22k_in1k':
            self.pre_model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True)
        elif args.model_name == 'convnextv2_base_22k':
            self.pre_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
        elif args.model_name == 'convnextv2_large_22k':
            self.pre_model = ConvNextV2ForImageClassification.from_pretrained("convnext/convnextv2-large-22k-224")
        elif args.model_name == 'eva':
            self.pre_model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)
            
        self.pre_model.eval()
        for pram in self.pre_model.parameters():
            pram.requires_grad = False
        
        self.M = torch.ones(channel_out, input_size, input_size, requires_grad=False, device=device)
        self.H_start = input_size // 2 - patch_H_size // 2
        self.H_end = self.H_start + patch_H_size        
        self.W_start = input_size // 2 - patch_W_size // 2
        self.W_end = self.W_start + patch_W_size
        self.M[:,self.H_start:self.H_end,self.W_start:self.W_end] = 0
        
        self.W = Parameter(torch.randn(channel_out, input_size, input_size, requires_grad=True, device=device))
        self.new_layers = nn.Sequential(nn.Linear(1000, 10))  ## Change to 200 when training TinyImagenet  
    
    def forward(self, image):
        X = torch.zeros(image.shape[0], self.channel_out, self.input_size, self.input_size)
        X[:,:,self.H_start:self.H_end,self.W_start:self.W_end] = image.repeat(1,1,1,1).data.clone()
        X = Parameter(X, requires_grad=True).to(self.device)

        P = torch.tanh(self.W * self.M)
        X_adv = P + X
        Y = self.pre_model(X_adv)
        Y = self.new_layers(Y)
        return Y

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def membership_inference_attack(model, train_loader, test_loader):
    model.eval()

    result = []
    softmax = torch.nn.Softmax(dim=1)

    train_cnt = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _y = softmax( model(x) )
        train_cnt += len(y)
        for i in range(len(_y)):
            result.append( [_y[i][y[i]].item(), 1] )

    test_cnt = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _y = softmax( model(x) )
        test_cnt += len(y)
        for i in range(len(_y)):
            result.append( [_y[i][y[i]].item(), 0] )

    result = np.array(result)
    result = result[result[:,0].argsort()]
    one = train_cnt
    zero = test_cnt
    best_atk_acc = 0.0
    for i in range(len(result)):
        atk_acc = 0.5 * (one/train_cnt + (test_cnt-zero)/test_cnt)
        best_atk_acc = max(best_atk_acc, atk_acc)
        if result[i][1] == 1:
            one = one-1
        else: zero = zero-1

    return best_atk_acc

device = 'cuda'

def main():
    parser = argparse.ArgumentParser(description='pate train')
    parser.add_argument('--seed', type=int, default=8872574) 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--size', type=int, default=192)
    #parser.add_argument('--target_size', type=int, default=196)  # # only EVA use this 
    parser.add_argument('--model_name', type=str, default='wideresnet',
                       choices=['wideresnet', 'resnet50', 'resnet152', 'swin', 'vit', 'vit_21k', 'swin_22k', 'swinv2_22k', 'swinv2_22k_ft_1k', 'swinv2_large_22k', 'convnextv2_ft_in22k_in1k', 'convnextv2_base_22k', 'convnextv2_large_22k', 'eva'])

    args = parser.parse_args()
    set_seed(args.seed)
    model = reProgrammingNetwork(args, patch_H_size=args.size, patch_W_size=args.size,device=device).to(device)
        
    checkpoint = torch.load('../AT_wideresnet/epoch19.pt')  # trained model checkpoint
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # setup data loader
    if args.dataset == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),        
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        test_transform = transforms.Compose([
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])        
        train_dataset = datasets.CIFAR10('../CIFAR10', train=True, transform=train_transform, download=True, )        
        test_dataset = datasets.CIFAR10('=../CIFAR10', train=False, transform=test_transform, download=True, )
    
    if args.dataset == 'TinyImagenet':
        train_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
        test_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),        
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        test_transform = transforms.Compose([
            transforms.Resize(args.size),             # resize shortest side to 224 pixels
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # dataset of TinyImageNet
        class TinyImageNetHuggingFace(Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]
                image = sample['image']
                label = sample['label']
                # RGB image
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label

        train_dataset = TinyImageNetHuggingFace(train_dataset, transform=train_transform)
        test_dataset = TinyImageNetHuggingFace(test_dataset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False) 
    atk_acc = membership_inference_attack(model, train_loader, test_loader)

    save_dir = '../AT_wideresnet/epoch19.pt'   # output MIA results
    log_file_path = os.path.join(save_dir, 'evaluation_miast.log')
    with open(log_file_path, 'a') as logfile:
        logfile.write(f"atk_acc:\n")
        logfile.write(f"{atk_acc}\n")
        logfile.write("\n")  

if __name__ == "__main__":
    main()