from datetime import datetime
from torch import nn, optim
import torch
from model import generator, discriminator
from data import load_mnist_dataloader
from argparse import ArgumentParser
import logging
import os
import json
import matplotlib.pyplot as plt
time = str(datetime.now().isoformat(timespec='minutes')).replace(':', '-')

def generate_and_save_images(model, epoch, fixed_noise, save_directory):
    model.eval()
    predictions = model(fixed_noise)
    predictions = predictions.detach().cpu()
    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(predictions[i, 0, :, :] * 127.5 + 127.5, cmap='gray')  # 去标准化
        plt.axis('off')  # 关闭坐标轴
    os.makedirs(f'{save_directory}/{time}/generate_imgs', exist_ok = True)
    # plt.savefig(f'{save_directory}/{time}/generate_imgs/epoch_{epoch}.png')
    plt.show()

def train_epoch(netG, netD, train_loader, noise_size:list, optimizerG, optimizerD, criterion, epoch, device):
    loss_generator = 0
    loss_discriminator = 0
    loss_total = 0
    netG.train()
    netD.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # 先训练判别器
        optimizerD.zero_grad()
        predict_real = netD(data)
        loss_real = criterion(predict_real, torch.full_like(predict_real, 0.9))
        noise1 = torch.randn(size=noise_size, device=device)
        fake1 = netG(noise1)
        predict_fake = netD(fake1)
        loss_fake = criterion(predict_fake, torch.full_like(predict_fake, 0.1))
        loss_D = loss_fake + loss_real
        loss_D.backward()
        optimizerD.step()
        loss_discriminator += loss_D.item()
        # 再训练生成器，一轮训练两次
        for _ in range(2):
            optimizerG.zero_grad()
            noise2 = torch.randn(size=noise_size, device=device)
            fake2 = netG(noise2)
            predict = netD(fake2)
            loss_G = criterion(predict, torch.full_like(predict, 0.9))
            loss_G.backward()
            optimizerG.step()
            loss_generator += loss_G.item()
        # 计算平均损失
        loss_generator_avg = loss_generator / ((batch_idx + 1) * 2)
        loss_discriminator_avg = loss_discriminator / (batch_idx + 1)
        # loss_total_avg = loss_total / (batch_idx + 1)
        if batch_idx % 200 == 0:
            logging.info(f'epoch:{epoch}\t[{batch_idx + 1}/{len(train_loader)}]\tloss_G:{loss_generator_avg}\tloss_D:{loss_discriminator_avg}')


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    netG = generator(args.z_dim).to(device)
    netD = discriminator().to(device)
    optimizerG = optim.Adam(params=netG.parameters(), lr=args.learning_rate_G, betas=args.betas)
    optimizerD = optim.Adam(params=netD.parameters(), lr=args.learning_rate_D, betas=args.betas)
    train_loader, _ = load_mnist_dataloader(args.batch_size)
    noise_size = [args.batch_size, args.z_dim, 1, 1]
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    fixed_noise = torch.randn(size=[36, args.z_dim, 1, 1], device=device)
    for epoch in range(1, args.epochs + 1):
        train_epoch(netG, netD, train_loader, noise_size, optimizerG, optimizerD, criterion, epoch, device)
        generate_and_save_images(netG, epoch, fixed_noise, args.save_dir)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate_G', type=float, default=0.0002)
    parser.add_argument('--learning_rate_D', type=float, default=0.0002)
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999))
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(f'{args.save_dir}/{time}', exist_ok = True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'./experiments/{time}/training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 将Namespace转为字典
    args_dict = vars(args)
    # 保存到文件
    with open(f'{args.save_dir}/{time}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=4)  # indent使文件可读

    main(args)