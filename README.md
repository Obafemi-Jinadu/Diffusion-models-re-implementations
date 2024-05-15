<h1 align="center">Re-implementation of the De-noised Diffusion Probabilistic Model (DDPM)</h1>

[DDPM](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf&ved=2ahUKEwj_t6yIs46GAxUUFVkFHV8RCssQFnoECBMQAQ&usg=AOvVaw3_txjfhqsg67acjkwqOuSf) [1] is a conditionless generative model which means the data generation is not guided or conditioned by anything like a text or a class. It randomly generates new data based on the training data distribution.

The diffusion model is a generative model that learns to generate data as follows, given training data:
- The model adds Gaussian noise at incremental steps by a 1st-order Markov chain, the increments are defined by a non-learnable diffusion rate $\beta$ that linearly increases by a simple linear noise scheduler (or a cosine noise scheduler is proposed in the improved DDPM). This is called the forward process, given by $q(x_{1:T}|x_{0})$. 
- The reverse process involves the model learning to iteratively de-noise random Gaussian samples until pristine data is generated. This is done by learning to predict the noise or Gaussian transitions at each time step from timestep t = T to t =0 given by $p_{\theta}(x_{0:T})$

On reparameterization, the forward process is efficiently derived as:
## Implementation Highlights
- Linear noise scheduler was used.
- Training data was the [Stanford cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) dataset, it was resized to a low resolution of 64 by 64 for faster training.
- Timesteps T = 300 and T = 1,000 was used.
- The simple U-Net architecture used to predict noise in the reverse process was adopted from this [work](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=597s).

Model generated image sample with timesteps, T = 300 arrow shows the transition from 300 to 0
 <h1 align="center"><img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/4caeeaf9560c278babd95e5527795a6c49139a14/files/arrow.png" width="160"/> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a007590f9335c0b0ac661cfea26deaf805ca2c03/files/img1.png" width="350"/></h1>


Model generated image sample with timesteps, T = 1,000 arrow shows the transition from 1,000 to 0 (Note: transition starts at T-100)
 <h1 align="center"> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/4caeeaf9560c278babd95e5527795a6c49139a14/files/arrow.png" width="160"/> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a8355896ea8e49c483e8fcf5ac31db31df38a122/files/img6.png" width="380"/></h1>

 


 


 
      
 1. write U-net code to predict noise
2. clean up train code
 3. Use argparse to take in image folder dir, batch size and more as input arguments when run on terminal

    # Note:
    `forward_process.py` just shows forward process on single image, code is unoptimized for `batch_size>1`. The optimized code has been written in `train.py`
