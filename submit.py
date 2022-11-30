import torch
import torch.nn as nn
import os
import zipfile
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import NonLocalFFC

import albumentations as A
import argparse

from utils import setup_seed
from dataset import build_dataset
from tqdm import tqdm
from skimage import io
from pathlib import Path

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='../data')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-0.pth')
    args = parser.parse_args()

    setup_seed(args.seed)

    train_dataloader, test_dataloader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NonLocalFFC(hidden_dim = 64)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    pred_img_list = []
    name_list = []

    with torch.no_grad():
        for img, file_name in tqdm(test_dataloader):
            img = img.to(device)
            predicted_img = model(img).to('cpu')
            for pred, name in zip(predicted_img, file_name):
                pred = pred.cpu().clone().detach()
                pred = pred.numpy()
                pred = pred.transpose(1, 2, 0)
                pred_img_list.append(pred.astype('uint8'))
                name_list.append(name)    
    sub_imgs = []
    Path.mkdir(Path("./submission"), exist_ok=True)
    os.chdir("./submission/")
    for path, pred_img in tqdm(zip(name_list, pred_img_list)):
        io.imsave(path, pred_img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("../submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('Done.')