<h1 align="center">Re-implementation of the De-noised Diffusion Probabilistic Model (DDPM)</h1>


  <img src="https://github.com/Obafemi-Jinadu/Diffusion-models-re-implementations/blob/faf7f1e31169c2730c555ce263c76bcdc4273907/files/img1.png |
width=100" width="200"/>
# TODOs on `train.py`
      
 1. write U-net code to predict noise
2. clean up train code
 3. Use argparse to take in image folder dir, batch size and more as input arguments when run on terminal

    # Note:
    `forward_process.py` just shows forward process on single image, code is unoptimized for `batch_size>1`. The optimized code has been written in `train.py`
