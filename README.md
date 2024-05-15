<h1 align="center">Re-implementation of the De-noised Diffusion Probabilistic Model (DDPM)</h1>
[DDPM](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf&ved=2ahUKEwj_t6yIs46GAxUUFVkFHV8RCssQFnoECBMQAQ&usg=AOvVaw3_txjfhqsg67acjkwqOuSf)

Model generated image sample with timesteps, T = 300
 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/4caeeaf9560c278babd95e5527795a6c49139a14/files/arrow.png" width="160"/> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a007590f9335c0b0ac661cfea26deaf805ca2c03/files/img1.png" width="350"/></h1>


Model generated image sample with timesteps, T = 1,000
 <h1 align="center"> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/4caeeaf9560c278babd95e5527795a6c49139a14/files/arrow.png" width="160"/> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a8355896ea8e49c483e8fcf5ac31db31df38a122/files/img6.png" width="380"/></h1>


 


 
      
 1. write U-net code to predict noise
2. clean up train code
 3. Use argparse to take in image folder dir, batch size and more as input arguments when run on terminal

    # Note:
    `forward_process.py` just shows forward process on single image, code is unoptimized for `batch_size>1`. The optimized code has been written in `train.py`
