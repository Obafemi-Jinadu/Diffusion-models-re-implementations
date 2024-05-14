from model import SimpleUnet
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torchinfo import summary
import torchvision.transforms as T
import numpy as np
from torch.nn.functional import relu
import tqdm
from tqdm import tqdm
import math

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Get arguments, data paths in this case.')
parser.add_argument('--data_path', type=str,
                    help='path to dataset')
parser.add_argument('--timesteps',default=300, type=int,
                    help='timesteps')
parser.add_argument('--img_size',default=64, type=int,
                    help='image size pass as int')
parser.add_argument('--batch_size',default=128, type=int,
                    help='batch size')
parser.add_argument('--epochs',default=300, type=int,
                    help='training epochs')


args = parser.parse_args()

timesteps = args.timesteps
data_path = Path(args.data_path)
img_size = args.img_size
batch_size = args.batch_size
epochs = args.epochs


data_transforms = transforms.Compose([
                                        transforms.Resize((img_size,img_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Lambda (lambda im: 2*im -1)
                                       ])
device = 'cuda' 
train_data = datasets.ImageFolder(data_path, transform=data_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


betas = torch.linspace(0.0001, 0.02, timesteps)
cumm_prod = torch.cumprod((1.-betas),-1)



@torch.no_grad()
def forward_diffusion(images, timesteps=300):
    
    
    batch_size = images.shape[0]
    ind =torch.randint(0, timesteps, (batch_size,)).long()
    selected_cumm_prod  = torch.gather( cumm_prod, 0, ind)
    selected_cumm_prod = selected_cumm_prod.reshape((-1,1,1,1))
     
    eps = torch.randn(images.shape)
    noise_inj = torch.sqrt(selected_cumm_prod)*images + (torch.sqrt(1.-selected_cumm_prod)*eps) #gets final noise level
    
    return eps, noise_inj, ind

model = SimpleUnet()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(epochs):
    running_loss = 0
    for images, _ in tqdm(trainloader):
        optimizer.zero_grad()
        noise, noised_img, t_ind = forward_diffusion(images)
        noise_pred = model(noised_img.to(device),t_ind.to(device))    
        loss= criterion(noise_pred, noise.to(device))
        loss.backward()
        optimizer.step()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        running_loss += loss.item()
       
    print('epoch ',epoch ,running_loss/len(trainloader.dataset))

torch.save(model.state_dict(), 'weights.pth')
