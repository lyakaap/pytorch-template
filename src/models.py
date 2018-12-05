import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AbemaNet(nn.Module):

    def __init__(self, emb_dims, n_channels_list=(64, 128), n_classes=1,
                 emb_dropout=0.5, dropout_list=(0.5, 0.5)):
        super(AbemaNet, self).__init__()

        n_numeric_feats = 0
        self.classifier = MultiModalNN(emb_dims=emb_dims, n_numeric_feats=n_numeric_feats,
                                       n_channels_list=n_channels_list, n_classes=n_classes,
                                       emb_dropout=emb_dropout, dropout_list=dropout_list)

    def forward(self, *input):
        pass
