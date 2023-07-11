import torch



def scale_data(x, scale_factor):
    return x * scale_factor



def create_domain_labels (batch_size, patch_n):
    domain_labels = torch.zeros(2*batch_size*patch_n, dtype=torch.long).cuda()
    num_labels = 2 * batch_size * patch_n
    for i in range(num_labels):
        if i < batch_size * patch_n:
            domain_labels[i] = 0
        else:
            domain_labels[i] = 1
    return domain_labels


def denormalize_(image, norm_range_max, norm_range_min):
    image = image * (norm_range_max -
                    norm_range_min) + norm_range_min
    return image


def normalize_(x, s, e):
    x = (x - s)/(e-s)
    return x


def trunc(mat, trunc_max, trunc_min):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat


