import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import NonLocalFFC

import argparse
import math
from focal_frequency_loss import FocalFrequencyLoss
from utils import setup_seed
from dataset import build_dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--block_size', type=int, default=256)    
    parser.add_argument('--dataset', type=str, default='../data')
    args = parser.parse_args()

    setup_seed(args.seed)

    train_dataloader, val_dataloader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NonLocalFFC(hidden_dim = 64)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)
    ffl = FocalFrequencyLoss().to(device)
    step_count = 0
    optim.zero_grad()

    for e in range(args.total_epoch):
        losses = []
        for img, label in tqdm(train_dataloader):
            step_count += 1
            img = img.to(device)
            predicted_img = model(img)
            mse = torch.mean((inv_normalize(label) - inv_normalize(predicted_img.cpu())) ** 2).detach().cpu(), 
            loss = criterion(label.to(device), predicted_img) + ffl(label.to(device), predicted_img)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(mse[0].item())
            mse_avg = sum(losses) / len(losses)
            psnr_avg = 10 * math.log10(1 / mse_avg)
            if step_count % 10 == 0:
                tqdm.write(f"step: {step_count} / {len(train_dataloader)} \t|  PSNR loss: "+ str(psnr_avg) + "dB"
                      f"\t|  MSE loss: {mse_avg}")
        print(f'In epoch {e}, average training loss is {psnr_avg} dB.')

        ''' save model '''
        if e % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint-{e}.pth")
    torch.save(model.state_dict(), f"checkpoint-final.pth")