from torchvision import datasets
import os
import concurrent

def stl10(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    train_set = datasets.STL10(root=file_dir, split='train', download=True)
    val_set = datasets.STL10(root=file_dir, split='test', download=True)
    return train_set, val_set

def save_img(entry, file_dir, mode, idx):
    img, label = entry
    label = str(label)
    if not os.path.exists(os.path.join(file_dir, mode, label)):
        os.makedirs(os.path.join(file_dir, mode, label))
    img_file = os.path.join(file_dir, mode, label, f'{idx}.jpg')
    img.save(img_file)
    return


if __name__ == '__main__':
    stl_dir = '/Volumes/INTEL/STL10'
    train_set, val_set = stl10(stl_dir)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(save_img, train_set, [stl_dir]*len(train_set), ['train']*len(train_set), range(len(train_set)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(save_img, val_set, [stl_dir]*len(val_set), ['val']*len(val_set), range(len(val_set)))