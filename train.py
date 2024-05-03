from torchvision import datasets, transforms, models

data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.Resize((224,224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Lambda (lambda im: 2*im -1)
                                       ])
cuda = torch.device('cuda')  
train_data = datasets.ImageFolder("", transform=data_transforms) # pass in link to dataset
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    
timesteps = 1000
betas = torch.linspace(0.0001, 0.02, timesteps)
cumm_prod = torch.cumprod((1.-betas),-1)
cumm_prod_sqrt = torch.sqrt(cumm_prod)
cumm_prod_sqrt_minus_one = torch.sqrt(1.-cumm_prod)


def forward_diffusion(images, t = -1):
    eps = torch.randn(images.shape)
    noise_inj = cumm_prod_sqrt[t]*images + (cumm_prod_sqrt_minus_one[t]*eps) #gets final noise level
    
    return noise_inj

for images, _ in trainloader:
    
    for t in range(timesteps-1, -1,-1):
        noised_img = forward_diffusion(images,t)
        noise_pred = model(noised_im, t)
        noise_added = noise_img - images
        loss= ...# func (noise_pred, noise_added )

# TODOs
      
# 1. write U-net code to predict noise
# 2. clean up train code
# 3. Use argparse
