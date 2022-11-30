import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from skimage import io
from skimage.transform import rescale
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, df, transforms, train_mode, dataset):
        self.df = df
        self.transforms = transforms
        self.train_mode = train_mode
        self.dataset = dataset

    def __getitem__(self, index):
        lr_path = str(Path(self.dataset)/ self.df['LR'].iloc[index])
        lr_img = io.imread(lr_path)
        lr_img = rescale(lr_img, 4, preserve_range=True,channel_axis=2).astype(np.uint8)
        if self.train_mode:
            hr_path = str(Path(self.dataset)/ self.df['HR'].iloc[index])
            hr_img = io.imread(hr_path)
            if self.transforms is not None:
                transformed = self.transforms(image=lr_img, label=hr_img)
                lr_img = transformed['image']
                hr_img = transformed['label']
            return lr_img, hr_img
        else:
            file_name = Path(lr_path).name
            if self.transforms is not None:
                transformed = self.transforms(image=lr_img)
                lr_img = transformed['image']
            return lr_img, file_name
        
    def __len__(self):
        return len(self.df)

def get_train_transform(args):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomCrop(args.block_size, args.block_size),
        A.Normalize((0, 0, 0), (1, 1, 1)),
        A.Normalize(max_pixel_value=1),
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )

def get_test_transform():
    return A.Compose([
        A.Normalize((0, 0, 0), (1, 1, 1)),
        A.Normalize(max_pixel_value=1),
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )

def build_dataset(args):
    train_df = pd.read_csv(str(Path(args.dataset)/ "train.csv"))
    test_df = pd.read_csv(str(Path(args.dataset)/ "test.csv"))
    train_dataset = CustomDataset(train_df, get_train_transform(args), True, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)

    test_dataset = CustomDataset(test_df, get_test_transform(), False, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    return train_loader, test_loader