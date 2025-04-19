import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        model += [nn.Conv2d(num_out, 1, 4, 1, 1)]

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


class SegNetCNN(nn.Module):
    """
    Basic implementation of SegNet for image segmentation.

    Args:
        in_channels (int): Number of input channels (e.g. 3 for RGB).
        out_channels (int): Number of output classes (e.g. 1 for binary segmentation).
        init_weights (bool): Whether to initialize weights using Kaiming initialization.
        nf (int): Base number of feature channels (default is 64).


    Loss function should match out_channels value:

        out_channels = 1 : BCEWithLogitsLoss()
        out_channels > 2 : CrossEntropyLoss()

    """
    def __init__(self, in_channels=3, out_channels=1, init_weights=True, nf=64, apply_softmax=False):
        super(SegNetCNN, self).__init__()

        self.apply_softmax = apply_softmax

        # Encoder
        self.enc1 = self.EncoderBlock(in_channels, nf, 2)
        self.enc2 = self.EncoderBlock(nf, nf*2, 2)
        self.enc3 = self.EncoderBlock(nf*2, nf*4, 3)
        self.enc4 = self.EncoderBlock(nf*4, nf*8, 3)
        self.enc5 = self.EncoderBlock(nf*8, nf*8, 3)

        #Decoder
        self.dec5 = self.DecoderBlock(nf*8, nf*8, 3)
        self.dec4 = self.DecoderBlock(nf*8, nf*4, 3)
        self.dec3 = self.DecoderBlock(nf*4, nf*2, 3)
        self.dec2 = self.DecoderBlock(nf*2, nf, 2)
        self.dec1 = self.DecoderBlock(nf, nf, 2)
        
        self.classifier = nn.Conv2d(nf, out_channels, kernel_size=1)

        


        if init_weights:
            self.init_weights()


           
    def forward(self,x):
        """
        Forward pass through the SegNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, out_channels, H, W).
        """   
        # Encoder
        x, idx1, size1 = self.enc1(x) 
        x, idx2, size2 = self.enc2(x)
        x, idx3, size3 = self.enc3(x)
        x, idx4, size4 = self.enc4(x)
        x, idx5, size5 = self.enc5(x)
       
        # Decoder
        x = self.dec5(x, idx5, size5)
        x = self.dec4(x, idx4, size4)
        x = self.dec3(x, idx3, size3)
        x = self.dec2(x, idx2, size2)
        x = self.dec1(x, idx1, size1)

        x = self.classifier(x)

        if self.apply_softmax:
            x = F.softmax(x, dim=1)

        return x
    
    def init_weights(self):
        """
        Initialize the weights of the model using Kaiming Normal initialization.

        This is typically used for layers with ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    

    class EncoderBlock(nn.Module):
        """
        Encoder block consisting of multiple convolutional layers followed by a max-pooling layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_conv (int): Number of convolutional layers in the block.
        """
        def __init__(self, in_channels, out_channels, num_conv):
            super().__init__()
            layers = []
            for _ in range(num_conv):
                layers += [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]

                in_channels = out_channels

            self.conv = nn.Sequential(*layers)
            self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        def forward(self, x):
            """
            Forward pass through the encoder block.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                tuple: Output tensor, pooling indices, and the size of the feature map.
            """
            x = self.conv(x)
            size = x.size()
            x, indices = self.pool(x)
            return x, indices, size

    class DecoderBlock(nn.Module):
        """
        Decoder block consisting of multiple convolutional layers followed by a max-unpooling layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_conv (int): Number of convolutional layers in the block.
        """
        def __init__(self, in_channels, out_channels, num_conv):
            super().__init__()
            layers = []
            for _ in range(num_conv):
                layers += [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ]
                in_channels = out_channels

            self.conv = nn.Sequential(*layers)
            self.unpool = nn.MaxUnpool2d(2, stride=2)

        def forward(self, x, indices, size):
            """
            Forward pass through the decoder block with unpooling.

            Args:
                x (torch.Tensor): Input tensor.
                indices (torch.Tensor): Indices from the corresponding encoder's pooling layer.
                size (torch.Size): The original size of the feature map from the encoder.

            Returns:
                torch.Tensor: The output tensor after unpooling and convolution.
            """
            x = self.unpool(x, indices, size)
            x = self.conv(x)
            return x
            





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