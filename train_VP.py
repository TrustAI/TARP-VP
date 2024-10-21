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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# save dir
model_dir = './Trained_Models/CIFAR10/VP_ResNet18/AT_resnet50'   
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# log file
log_file = os.path.join(model_dir, 'training.log')
log_format = ""
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=8/255,
                  num_steps=10,
                  step_size=2/255):
    model.eval()

    X_pgd = Variable(X.data, requires_grad=True)
    
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    opt = optim.SGD([X_pgd], lr=1e-3)

    for _ in range(num_steps):
        
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def train_model(dataset, test_dataset, args):
    device = args.device
    num_epochs = args.epoch
    batch_size = args.batch_size
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True) 
    # add input_size=args.target_size when using EVA model
    model = reProgrammingNetwork(args, patch_H_size=args.size, patch_W_size=args.size,device=device).to(device)
    loss_function = nn.CrossEntropyLoss()

    # Using ResNet18 or WideResNet to generate AEs in test phase
    model2 = torchvision.models.resnet18().to(device)  
    # model2 = WideResNet().to(device)

    # SAM optimizer
    base_optimizer = optim.SGD
    optimizer = SAM(filter(lambda p: p.requires_grad, model.parameters()), base_optimizer, lr = args.lr, momentum = 0.9)
    scheduler = StepLR(optimizer, args.LR_step, gamma=args.gamma)
    best_test_acc = 0;
    end_train_acc = 0;
    best_test_adv = 0;
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0
        train_acc = 0
        train_adv = 0
        test_acc = 0
        test_adv = 0
        for i, (image, label) in enumerate(tqdm(trainloader)):

            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                X, y = Variable(image, requires_grad=True), Variable(label)
                image_adv = _pgd_whitebox(copy.deepcopy(model2), X, y)

            label_hat = model(image)
            label_adv = model(image_adv)
            
            if args.mode == 'ST':
                loss = loss_function(label_hat, label)  # ST loss
            if args.mode == 'AT':
                loss = loss_function(label_adv, label)  # PGDAT loss
            train_loss += loss.item()
            train_acc += sum(label.cpu().numpy() == label_hat.data.cpu().numpy().argmax(1))
            train_adv += sum(label.cpu().numpy() == label_adv.data.cpu().numpy().argmax(1))
            loss.backward()

            optimizer.first_step(zero_grad=True)
            
            if args.mode == 'ST':
                loss_function(model(image), label).backward()  # ST loss
            if args.mode == 'AT':
                loss_function(model(image_adv), label).backward() # PGDAT loss
            optimizer.second_step(zero_grad=True)
            
        # test
        model.eval()
        with torch.no_grad():
            for image, label in testloader:   
                image, label = image.to(device), label.to(device)

                with torch.no_grad():
                    X, y = Variable(image, requires_grad=True), Variable(label)
                    image_adv = _pgd_whitebox(copy.deepcopy(model2), X, y, num_steps=20)

                preds = model(image).data.cpu().numpy().argmax(1)
                test_acc += sum(label.cpu().numpy() == preds)
                preds_adv = model(image_adv).data.cpu().numpy().argmax(1)
                test_adv += sum(label.cpu().numpy() == preds_adv)
                
        testacc = test_acc/float(len(test_dataset))
        testadv = test_adv/float(len(test_dataset))

        print("Test Accuracy: {:.4f}".format(testacc))
        print("Test Adv Accuracy: {:.4f}".format(testadv))
        if testacc > best_test_acc:
            best_test_acc = testacc
        if testadv > best_test_adv:
            best_test_adv = testadv            
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))
                    
        scheduler.step()
        end_train_acc = train_acc/len(dataset)
        print("Epoch: {}".format(epoch+1),
              "Train Loss: {:.3f}".format(train_loss/len(trainloader)),
              "Train Accuracy: {:.4f}".format(train_acc/len(dataset)),
              "Train Adv Accuracy: {:.4f}".format(train_adv/len(dataset)),
              "lr: {}".format(scheduler.get_last_lr()[0]))  

        logging.info("======= Epoch {} =======".format(epoch + 1))
        logging.info("Loss: {:.4f}.	LR: {:.4f}".format(loss, scheduler.get_last_lr()[0]))
        logging.info("Standard Accuracy-	Train: {:.4f}.	Test: {:.4f}.".format(train_acc/len(dataset), testacc))
        logging.info("Adversarial Accuracy-	Train: {:.4f}.	Test: {:.4f}.".format(train_adv/len(dataset), testadv))
        logging.info("Time taken: {:.2f} seconds".format(time.time() - start))

    # save last epoch   
    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))
    return end_train_acc, best_test_acc, best_test_adv, model

def main():
    parser = argparse.ArgumentParser(description='pate train')
    parser.add_argument('--seed', type=int, default=8872574) #4
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.05) # or 0.01 if the training is poor (eg, ResNet50, Wideresnet for TinyImagenet)
    parser.add_argument('--LR_step', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--mode', type=str, default='AT')
    #parser.add_argument('--target_size', type=int, default=196)  # only EVA use this 
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['wideresnet', 'resnet50', 'resnet152', 'swin', 'vit', 'vit_21k', 'swin_22k', 'swinv2_22k', 'swinv2_22k_ft_1k', 'swinv2_large_22k', 'convnextv2_ft_in22k_in1k', 'convnextv2_base_22k', 'convnextv2_large_22k', 'eva'])

    args = parser.parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {args.device} backend")

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
   
    # train
    train_acc, test_acc, test_adv, st_model = train_model(train_dataset, test_dataset, args)
    
    print("###########end_train:", train_acc, "," , "best_test_acc_adv:", test_acc, "," , test_adv, "###########")
    
if __name__ == "__main__":
    main()