import torch
import torch.nn as nn
from Unetmamba import rest_lite, MambaSegDecoder,UNetMamba

class RefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))

class SARRefinementModel(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 decode_channels: int = 64,
                 backbone_path: str = 'pretrain_weights/rest_lite.pth',
                 embed_dim: int = 64,
                 in_chans: int = 1,
                 num_classes: int = 1,
                 use_lsm: bool = False,
                 **kwargs
                 ):
        super().__init__()

        self.encoder = rest_lite(
            weight_path=backbone_path,
            in_chans=in_chans,
            pretrained=pretrained,
            **kwargs
        )

        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.decoder = MambaSegDecoder(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            use_lsm=use_lsm
        )

        self.input_proj = nn.Sequential(
            nn.Conv2d(num_classes + in_chans, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.refine_layers = nn.Sequential(
            RefineBlock(32),
            RefineBlock(32)
        )
        
        self.output_proj = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, x):

        h, w = x.size()[-2:]

        enc_features = self.encoder(x)
        if hasattr(self.decoder, 'use_lsm') and self.decoder.use_lsm and self.training:

             umambaSAR1, lsm = self.decoder(enc_features, h, w)
        else:
             umambaSAR1 = self.decoder(enc_features, h, w)
             lsm = None

        combined = torch.cat([umambaSAR1, x], dim=1)
        

        feat = self.input_proj(combined)
        feat = self.refine_layers(feat)
        residual = self.output_proj(feat)

        finalSAR = umambaSAR1 + residual
        
        if self.training:
            return umambaSAR1, finalSAR, lsm
        else:
            return finalSAR
