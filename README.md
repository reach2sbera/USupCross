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

# Dataset Preparation
Please refer to the prep.py file to prepare the raw DICOM into training data. Our data loader necessitates pre-processed and normalized data in '.npy' format. The Source domain data (i.e., paired data) should be organized in a folder, where the LDCT image has a filename ending with '__input.npy' and the corresponding NDCT image has a filename ending with '__target.npy'. The Target domain LDCT images should also adhere to a similar naming syntax. 
