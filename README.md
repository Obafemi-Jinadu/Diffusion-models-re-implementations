<h1 align="center">Re-implementation of the De-noised Diffusion Probabilistic Model (DDPM)</h1>

Model output, 300 timesteps
 <h1 align="center"> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a007590f9335c0b0ac661cfea26deaf805ca2c03/files/img1.png" width="350"/></h1>


 Model output, 1000 timesteps
 <h1 align="center"> <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a8355896ea8e49c483e8fcf5ac31db31df38a122/files/img6.png" width="350"/></h1>


 Model output, 300 timesteps

 <p float="left">
  <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a007590f9335c0b0ac661cfea26deaf805ca2c03/files/img1.png" width="350"/>
  <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a8355896ea8e49c483e8fcf5ac31db31df38a122/files/img6.png" width="380"/>
</p>


 Model output, 300 timesteps             |  Model output, 1,000 timesteps
:-------------------------:|:-------------------------:

label 1 | label 2
--- | ---
<img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a007590f9335c0b0ac661cfea26deaf805ca2c03/files/img1.png" width="350"/>  |  <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/a8355896ea8e49c483e8fcf5ac31db31df38a122/files/img6.png" width="380"/>




    <img src="https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/de2effb13ab47f7ee64cb52602c24861f19106b6/multi%20media%20files/city%20street_clear_a85cad42-337048b5.jpg" width="200"/>
# TODOs on `train.py`
      
 1. write U-net code to predict noise
2. clean up train code
 3. Use argparse to take in image folder dir, batch size and more as input arguments when run on terminal

    # Note:
    `forward_process.py` just shows forward process on single image, code is unoptimized for `batch_size>1`. The optimized code has been written in `train.py`
