import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, ch_first=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            ch_first (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'deconv' or 'upconv'.
                           'deconv' will use transposed convolutions for
                           learned upsampling.
                           'upconv' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('deconv', 'upconv')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (ch_first + i),
                                                padding, batch_norm))
            prev_channels = 2**(ch_first + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (ch_first + i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(ch_first + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upconv':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class MultiModalNN(nn.Module):
    # https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

    def __init__(self, emb_dims, n_numeric_feats, n_channels_list=(64, 128),
                 n_classes=1, emb_dropout=0.2, dropout_list=(0.5, 0.5)):

        """
        Parameters
        ----------

        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        n_numeric_feats: Integer
          The number of continuous features in the data.

        n_channels_list: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        n_classes: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        dropout_list: List of floats
          The dropouts to be used after each linear layer.

        Examples
        --------
        >>> cat_dims = [int(data[col].nunique()) for col in categorical_features]
        >>> cat_dims
        [15, 5, 2, 4, 112]
        >>> emb_dims = [(x, min(32, (x + 1) // 2)) for x in cat_dims]
        >>> emb_dims
        [(15, 8), (5, 3), (2, 1), (4, 2), (112, 32)]
        >>> model = MultiModalNN(emb_dims, n_numeric_feats=4, lin_layer_sizes=[50, 100],
        >>>                      output_size=1, emb_dropout=0.04,
        >>>                      lin_layer_dropouts=[0.001,0.01]).to(device)
        """

        super(MultiModalNN, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.n_numeric_feats = n_numeric_feats

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.n_numeric_feats, n_channels_list[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [
            nn.Linear(n_channels_list[i], n_channels_list[i + 1]) for i in range(len(n_channels_list) - 1)
        ])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(n_channels_list[-1], n_classes)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.n_numeric_feats)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in n_channels_list])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in dropout_list])

    def forward(self, numeric_feats, categorical_feats):

        if self.no_of_embs != 0:
            x = [emb_layer(categorical_feats[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.n_numeric_feats != 0:
            normalized_numeric_feats = self.first_bn_layer(numeric_feats)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_numeric_feats], 1)
            else:
                x = normalized_numeric_feats

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x
