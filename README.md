<h1 align="center">Re-implementation of the De-noised Diffusion Probabilistic Model (DDPM)</h1>

![image](https://github.com/Obafemi-Jinadu/Survey-on-Low-Light-Image-Enhancement-with-Deep-learning/blob/2620d56bfda48ccbe25c877942c4280b9a62f222/multi%20media%20files/final_img.png)
# TODOs on `train.py`
      
 1. write U-net code to predict noise
2. clean up train code
 3. Use argparse to take in image folder dir, batch size and more as input arguments when run on terminal

    # Note:
    `forward_process.py` just shows forward process on single image, code is unoptimized for `batch_size>1`. The optimized code has been written in `train.py`
