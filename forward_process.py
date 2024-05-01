import numpy as np
from matplotlib import pyplot as plt
import torch

def forward_process(time_steps, img):
    comp_iter =[]
    betas = torch.linspace(0.0001, 0.02, time_steps)
    cumm_prod = 1
    eps = torch.randn(img.shape)
    for i, beta in enumerate(betas):
        cumm_prod *= (1-beta)
        noise_inj = ((cumm_prod)**0.5)*img + ((1-cumm_prod)**0.5)*eps
        comp_iter.append(noise_inj)
    
    return comp_iter

#  Visualize noise injection
lis = forward_process(1000, im)

plt_1 = plt.figure(figsize=(15, 15))
for i in range(0, 1001, 100):
    if i ==0:
        plt.subplot(3, 5, i+1)
        plt.imshow(lis[i])
        plt.title(f'iteration {i}')
    else: 
        plt.subplot(3, 5, i//100)
        plt.imshow(lis[i-1])
        plt.title(f'iteration {i-1}')
        
