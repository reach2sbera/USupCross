# USupCross
Official Code for the paper Unsupervised Domain Adaptive Cross Consistency Training for Out of Distribution Low Dose Computed Tomography Image Denoising

# Code Dependencies

The following dependencies are required to run the code:

- Python 3.x
- PyTorch
- torch-vision
- NumPy

# Usuage

python train.py --mode train --load_mode 0 --save_path 'your_path' --path_ldct 'path to source domain data' --path_real 'path to target domain data' --scale_data False --transform False --patch_n 3 --patch_size 100 --batch_size 3 --num_epochs 200 --print_iters 20 --decay_iters 6000 --save_iters 1000 --test_iters 1000 --device your_device --lr 1e-5 --load_chkpt False --num_workers 7

# Trained Models
Trained models will be uploaded once the paper is accepted. 
