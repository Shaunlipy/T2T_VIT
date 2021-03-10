from pathlib import Path
import os
import glob

def glob_imgs(file_path, mode):
    imgs_ = glob.glob(os.path.join(file_path, f'{mode}/*/*.jpg'))
    imgs_.extend(glob.glob(os.path.join(file_path, f'{mode}/*/*.JPEG')))
    if 'Imagenet_resize' not in mode and 'LSUN_resize' not in mode:
        imgs_ = [f'{mode}/{Path(line).parent.stem}/{Path(line).name},{Path(line).parent.stem}' for line in imgs_]
    else:
        imgs_ = [f'{mode}/{Path(line).parent.stem}/{Path(line).name},0' for line in imgs_]
    return imgs_

if __name__ == '__main__':
    # LSUN_resize
    file_path = '/Volumes/CT500/Researches/Attention_OOD/data'
    val_imgs = glob_imgs(file_path, 'LSUN_resize')
    with open('data/LSUN_resize_ood.txt', 'w') as f:
        f.write('\n'.join(val_imgs))
    print(len(val_imgs))

    #Imagenet_resize
    # file_path = '/Volumes/CT500/Researches/Attention_OOD/data'
    # val_imgs = glob_imgs(file_path, 'Imagenet_resize')
    # with open('data/imagenet_resize_ood.txt', 'w') as f:
    #     f.write('\n'.join(val_imgs))
    # print(len(val_imgs))

    # SVHN
    # file_path = '/Volumes/INTEL/SVHN/'
    # val_imgs = glob_imgs(file_path, 'val')
    # with open('data/svhn_val.txt', 'w') as f:
    #     f.write('\n'.join(val_imgs))
    # print(len(val_imgs))

    # TinyImageNet
    # file_path = '/Volumes/INTEL/TinyImageNet/'
    # val_imgs = glob_imgs(file_path, 'test')
    # with open('data/tinyimagenet_test.txt', 'w') as f:
    #     f.write('\n'.join(val_imgs))
    # print(len(val_imgs))


    # CIFAR10
    # file_path = '/Volumes/INTEL/CIFAR10'
    # train_imgs = glob_imgs(file_path, 'train')
    # print(len(train_imgs))
    # with open('data/cifar10_train.txt', 'w') as f:
    #     f.write('\n'.join(train_imgs))
    # val_imgs = glob_imgs(file_path, 'test')
    # with open('data/cifar10_test.txt', 'w') as f:
    #     f.write('\n'.join(val_imgs))
    # print(len(val_imgs))


    # STL10
    # file_path = '/Volumes/INTEL/STL10'
    # train_imgs = glob_imgs(file_path, 'train')
    # print(len(train_imgs))
    # with open('data/stl_train.txt', 'w') as f:
    #     f.write('\n'.join(train_imgs))
    # val_imgs = glob_imgs(file_path, 'val')
    # with open('data/stl_val.txt', 'w') as f:
    #     f.write('\n'.join(val_imgs))
    # print(len(val_imgs))