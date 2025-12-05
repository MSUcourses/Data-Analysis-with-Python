import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        #ВАШ КОД ЗДЕСЬ
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        #ВАШ КОД ЗДЕСЬ
    )

    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()


        # добавьте несколько слоев encoder block
        # это блоки-составляющие энкодер-части сети
        self.encoder = nn.Sequential(
            #ВАШ КОД ЗДЕСЬ
        )

        # добавьте несколько слоев decoder block
        # это блоки-составляющие декодер-части сети
        self.decoder = nn.Sequential(
            #ВАШ КОД ЗДЕСЬ
        )

    def forward(self, x):

        # downsampling 
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction



def create_model():
    return Autoencoder()
