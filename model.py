
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        """
        UNetBlock class represents a single block in the U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool, optional): Whether to perform downsampling or upsampling. Defaults to False.
        """
        super(UNetBlock, self).__init__()
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            relu_layer = nn.LeakyReLU(0.2, True)
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            relu_layer = nn.ReLU(True)
        
        norm_layer = nn.BatchNorm2d(out_channels)

        self.model = nn.Sequential(conv_layer, relu_layer, norm_layer)

    def forward(self, x):
        """
        Forward pass of the UNetBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class ShadowGenerator(nn.Module):
    """ 
    ShadowGenerator class represents the generator for shadow masks.

    Args:
        input_channels (int, optional): Number of input channels. Defaults to 3.
        output_channels (int, optional): Number of output channels. Defaults to 1.
        ngf (int, optional): Number of filters in the generator. Defaults to 64.
    """
    def __init__(self, input_channels=3, output_channels=1, ngf=64):
        super(ShadowGenerator, self).__init__()

        self.input = nn.Conv2d(input_channels, ngf, 4, 2, 1)

        # Encoder
        self.down1 = UNetBlock(ngf, ngf * 2, down=True)
        self.down2 = UNetBlock(ngf * 2, ngf * 4, down=True)
        self.down3 = UNetBlock(ngf * 4, ngf * 8, down=True)

        # Bottleneck
        self.bridge1 = UNetBlock(ngf*8, ngf*8, down=True)
        self.bridge2 = UNetBlock(ngf*8, ngf*8)

        # Decoder
        self.up3 = UNetBlock(ngf * 8 * 2, ngf * 4)
        self.up2 = UNetBlock(ngf * 4 * 2, ngf * 2)
        self.up1 = UNetBlock(ngf * 2 * 2, ngf)

        self.output = nn.ConvTranspose2d(ngf*2, output_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the ShadowGenerator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bridge1(x4)
        x6 = self.bridge2(x5)
        x7 = self.up3(torch.cat([x4, x6], 1))
        x8 = self.up2(torch.cat([x3, x7], 1))
        x9 = self.up1(torch.cat([x2, x8], 1))
        x10 = self.output(torch.cat([x1, x9], 1))
        return self.tanh(x10)

class ShadowDiscriminator(nn.Module):
    """ 
    Discriminator class represents the discriminator for shadow masks.

    Args:
        input_channels (int, optional): Number of input channels. Defaults to 4.
        ndf (int, optional): Number of filters in the discriminator. Defaults to 64.
        num_layers (int, optional): Number of layers in the discriminator. Defaults to 3.
    """
    def __init__(self, input_channels=4, ndf=64, num_layers=3):
        super(ShadowDiscriminator, self).__init__()

        model = [nn.Conv2d(input_channels, ndf, 4, 2, 1),
                 nn.LeakyReLU(0.2, True)]
        
        for i in range(0, num_layers):
            num_in = min(2**i * ndf, 512)
            num_out = min(2**(i+1) * ndf, 512)
            model += [nn.Conv2d(num_in, num_out, 4, 2, 1), 
                      nn.BatchNorm2d(num_out),
                      nn.LeakyReLU(0.2, True)
                    ]
        
        model += [nn.Conv2d(num_out, 1, 4, 1, 1), nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


def init_weights(net, init_gain=0.02):
    """
    Initialize the weights of the network.

    Args:
        net (torch.nn.Module): Network to initialize weights.
        init_gain (float, optional): Gain factor for initialization. Defaults to 0.02.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)



def test():
    model = ShadowGenerator(3, 1)
    x = torch.randn(1, 3, 256, 256)
    print("Size of input image: 256x256x3")
    print("Size of output image: 256x256x1")
    print(model(x).shape)

    model = ShadowDiscriminator()
    x = torch.randn(1, 4, 256, 256)
    print("Size of input image: 256x256x4")
    print("Size of output image: 256x256x1")
    print(model(x).shape)

# test()