import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from core.utils.diffskill_utils import img_to_tensor, batch_pred


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class CNNVAE(nn.Module):
    def __init__(self, args, image_channels=3, h_dim=1024, dimz=32, vae_beta=1.):
        """Taken from https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb"""
        super(CNNVAE, self).__init__()
        self.args = args
        self.h_dim = h_dim
        self.dimz = dimz
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, dimz)
        self.fc2 = nn.Linear(h_dim, dimz)
        self.fc3 = nn.Linear(dimz, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
        )
        self.vae_beta = vae_beta

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        # decode shape: BCHW
        z = self.fc3(z)
        z = self.decoder(z)
        z = torch.cat([nn.functional.sigmoid(z[:, :3, :, :]), z[:, 3:, :, :]], dim=1)
        return z

    def reconstr(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def sample_latents(self, n, device):
        z = torch.normal(0., 1., size=(n, self.dimz), device=device)
        return z

    def loss_fn(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        B = recon_x.shape[0]
        BCE = ((recon_x - x) ** 2).sum() / B

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

        return BCE + self.vae_beta * KLD, BCE, KLD

    def generate_cached_buffer(self, buffer):
        def encode(x):
            if not isinstance(x, torch.Tensor):
                x = img_to_tensor(x, mode='rgbd').to('cuda')
            z, _, _ = self.encode(x)
            return z.detach().cpu()

        obses = buffer.buffer['obses']
        self.cached_buffer_obs = batch_pred(encode, {'x': obses}, batch_size=512)
        self.cached_buffer_goal = batch_pred(encode, {'x': buffer.target_imgs}, batch_size=512)
        print('Cached buffer of length {} generated!'.format(len(self.cached_buffer_obs)))

    def get_cached_encoding(self, idx, noise=False, goal=False):
        if goal:
            z = self.cached_buffer_goal[idx]
        else:
            z = self.cached_buffer_obs[idx]
        if not noise:
            return z
        noise = torch.normal(mean=0, std=self.args.rgb_vae_noise, size=z.shape, device=z.device)
        return z + noise

    def sample_u(self, batch_size, truncate_std_latent=None, cache=False):
        """Wrapper to be compatible with pointflow VAE"""
        return self.sample_latents(batch_size, 'cuda')
