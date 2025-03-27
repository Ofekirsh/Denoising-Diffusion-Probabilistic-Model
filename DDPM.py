"""DDPM model for MNIST
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from UNet import Unet
class DDPM(nn.Module):
    def __init__(self, timestepstimesteps=1000, guidance=True,device='cpu'):
        """
        DDPM model for conditional MNIST generation
        :param timesteps: Number of timesteps in the generation
        :param guidance: Run with or without classifier free guidance.
        :param device: cpu or gpu
        """
        super(DDPM, self).__init__()
        self.device = device
        self.guidance = guidance
        self.timesteps = 1000
        self.in_channels = 1
        self.image_size = 28
        self.model=Unet(self.timesteps,64)
        self.betas = self._cosine_variance_schedule(self.timesteps).to(device)
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas,dim=-1)
        self.loss = nn.MSELoss()

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas


    def sample(self):
        '''

        :return: samples 100 images conditioned on y =torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        '''
        y = torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        x = torch.randn(100,1,28,28).to(self.device)
        uncond = True
        x_t = x
        for t in range(self.timesteps-1, -1, -1):
            # predict noise from the Unet
            t_tensor = torch.full((x_t.size(0), ), t, dtype=torch.long, device=self.device)
            if self.guidance:
                noise_condi = self.model(x_t, t_tensor, y, uncond=False)
                noise_wo_condi = self.model(x_t, t_tensor, uncond=True)
                noise = noise_wo_condi + 1.5 * (noise_condi - noise_wo_condi)
            else:
                noise = self.model(x_t, t_tensor, y, uncond=False)

            # Compute alpha and beta for the current timestep
            alpha_t = self.alphas[t].reshape(-1,1,1,1)
            alpha_bar_t = self.alpha_bars[t].reshape(-1,1,1,1)
            beta_t = self.betas[t].reshape(-1,1,1,1)

            mean = torch.sqrt(1.0/ alpha_t) * (x_t - noise * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t))
            if t > 0:
                std_for_last_one = torch.sqrt(beta_t) * torch.randn_like(x_t)
                x_t = mean + std_for_last_one
            else:
                x_t = mean

        return x_t.to('cpu')

    def forward(self, x,epsilon,t,y):
        '''
        Given a clean image x, random noise epsilon and time t, sample x_t and return the noise estimation given x_t
        :param x: Clean MNIST images
        :param epislon: i.i.d normal noise size of x
        :param t: time from 1 to time step
        :param y: labels
        :return: estimated_epsilon
        '''
        proba_drop_label = 0.15
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t).reshape(-1, 1, 1, 1)


        x_t = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * epsilon

        if not self.guidance:
            predicted_noise = self.model(x_t, t,y,uncond=False)
        else:
            predicted_noise = torch.zeros_like(x_t)
            mask = torch.rand(x_t.size(0), device=x_t.device) > proba_drop_label

            mask_condi = mask.nonzero().squeeze(1)
            mask_wo_condi = (~mask).nonzero().squeeze(1)

            if mask_condi.numel() > 0:
                predicted_noise[mask_condi] = self.model(x_t[mask_condi],
                                                         t[mask_condi],
                                                         y[mask_condi],
                                                         uncond=False)
            if mask_wo_condi.numel() > 0:
                predicted_noise[mask_wo_condi] = self.model(x_t[mask_wo_condi],
                                                         t[mask_wo_condi],
                                                         y[mask_wo_condi],
                                                         uncond=True)
        return predicted_noise