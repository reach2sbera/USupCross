import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    def __init__(self, mode, load_file, load_mode, saved_path, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"
        if load_file == 'input':
            _path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        else :
            _path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in _path ]
            # target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0: # batch data load
                self.input_ = input_
                # self.target_ = target_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                # self.target_ = [np.load(f) for f in target_]
        else: # mode =='test'
            input_ = [f for f in _path]
            # target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:
                self.input_ = input_
                # self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                # self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        input_img = self.input_[idx]
        if self.load_mode == 0:
            input_img = np.load(input_img)

        if self.transform:
            input_img = self.transform(input_img)

        if self.patch_size:
            input_patches = get_patch(input_img, self.patch_n, self.patch_size)
            return (input_patches)
        else:
            return (input_img)


def get_patch(full_input_img, patch_n, patch_size):
    # assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    # patch_target_imgs = []
    #print(full_input_img.shape)
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        # patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        # patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs)


def get_loader(mode='train', load_file='input', load_mode=0,
               saved_path=None,
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_file, load_mode, saved_path, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader
