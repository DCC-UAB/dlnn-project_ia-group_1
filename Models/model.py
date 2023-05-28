from einops import rearrange
import torch
import torch.nn as nn
import torchvision

class ConTextTransformer(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)          # Download the ResNet-50 architecture with pretrained weights.
        modules = list(resnet50.children())[:-2]                         # Keep all layers until the linear maps.
        self.resnet50 = nn.Sequential(*modules)                          # Convert ResNet-50 modules to a sequential object.
        for param in self.resnet50.parameters():                         # Freeze the weights to perform feature extraction.
            param.requires_grad = False
        self.num_cnn_features = 64                                       # Number of CNN features extracted (corresponds to an activation map of 8x8).
        self.dim_cnn_features = 2048                                     # Dimension of the last activation map of ResNet-50 (2048 channels).
        self.dim_fasttext_features = 300                                 # Dimension of the FastText features.

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim))  # Learnable positional embedding tensor with shape (1, num_cnn_features + 1, dim).
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)              # Linear transformation to project CNN features to the desired dimension.
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)    # Linear transformation to project FastText features to the desired dimension.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                              # Learnable CLS token with shape (1, 1, dim) for class prediction during training.
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)  # Define the encoder layer for the Transformer.
        #encoder_norm = nn.LayerNorm(dim)                                                                    # Layer normalization for the encoder output.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)                            # Transformer encoder to capture contextual information. 

        self.to_cls_token = nn.Identity()                                                 # Identity layer to extract the CLS token from the transformer output.

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),                                                      # MLP head to perform classification on the extracted features.
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)                      # Obtain the feature maps for the given image. Shape: (batch_size, 2048, H, W).
        x = rearrange(x, 'b d h w -> b (h w) d')    # Rearrange the tensor for compatibility with nn.Linear. Shape: (batch_size, (H x W), dim).
        x = self.cnn_feature_to_embedding(x)        # Apply a linear map to project the CNN features to the desired dimension.

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1) # Create a CLS token for each image in the batch. Shape: (batch_size, 1, dim).
        x = torch.cat((cls_tokens, x), dim=1)                    # Concatenate the CLS token with the feature maps.

        x += self.pos_embedding                                 # Add the positional embedding tensor to the concatenated tensor.

        x2 = self.fasttext_feature_to_embedding(txt.float())    # Apply a linear map to project the FastText features to the desired dimension.
        x = torch.cat((x, x2), dim=1)                           # Concatenate the transformed FastText features with the image features.

        #tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)  # Create a temporary mask tensor.
        #mask = torch.cat((tmp_mask.to(device), mask), dim=1)                               # Concatenate the temporary mask with the input mask.
        #x = self.transformer(x, src_key_padding_mask=mask)                                 # Apply the transformer encoder with the source key padding mask.
        x = self.transformer(x)                                                             # Apply the transformer encoder.

        x = self.to_cls_token(x[:, 0]) #The first token of the transformer output is extracted using x[:, 0]. Which corresponds to the cls token
        return self.mlp_head(x)  # Pass the extracted token through the MLP head for classification.