from dcgan import generator
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z1 = torch.randn(1, latent_dim)   # 形状 (1, latent_dim)
z2 = torch.randn(1, latent_dim)

generator = generator(latent_dim)
# generator.load_state_dict(torch.load('./experiments/2026-02-23T18-35/checkpoints/mnist-dcgan-epoch30.pth'))
generator.load_state_dict(torch.load('./mnist-dcgan-epoch30.pth'))
generator = generator.to(device)
generator.eval()

num_steps = 36
alphas = np.linspace(0, 1, num_steps)
interpolated_zs = []
for alpha in alphas:
    z = (1 - alpha) * z1 + alpha * z2
    interpolated_zs.append(z)
interpolated_zs = torch.cat(interpolated_zs, dim=0)
interpolated_zs = interpolated_zs.view(-1, latent_dim, 1, 1).to(device)
with torch.no_grad():
    predictions = generator(interpolated_zs)
    predictions = predictions.detach().cpu()
    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(predictions[i, 0, :, :] * 127.5 + 127.5, cmap='gray')  # 去标准化
        plt.axis('off')  # 关闭坐标轴
    # os.makedirs(f'{save_directory}/{time}/generate_imgs', exist_ok=True)
    # plt.savefig(f'{save_directory}/{time}/generate_imgs/epoch_{epoch}.png')
    plt.show()
