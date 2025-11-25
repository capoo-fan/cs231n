import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        N, T = captions.shape
        # Create a placeholder, to be overwritten by your code below.
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: Implement the forward function for CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have to embed your caption and add positional              #
        #     encoding. You then have to project the image features into the same  #
        #     dimensions.                                                          #
        #  2) You have to prepare a mask (tgt_mask) for masking out the future     #
        #     timesteps in captions. torch.tril() function might help in preparing #
        #     this mask.                                                           #
        #  3) Finally, apply the decoder features on the text & image embeddings   #
        #     along with the tgt_mask. Project the output to scores per token      #
        ############################################################################
        # 1. 文本嵌入与位置编码 (Embed Caption & Positional Encoding)
        # 将 caption token 索引转换为向量：(N, T) -> (N, T, W)
        caption_embed = self.embedding(captions)
        # 加入位置编码，赋予模型对序列顺序的感知能力
        caption_embed = self.positional_encoding(caption_embed)

        # 2. 图像特征投影 (Project Image Features)
        # 将图像特征从维度 D 投影到与文本向量相同的维度 W
        # features: (N, D) -> projected_features: (N, 1, W) -> (N, S=1, W)
        # 这里假设每个图像只有一个特征向量，所以源序列长度 S=1
        projected_features = self.visual_projection(features).unsqueeze(1)

        # 3. 构建目标序列掩码 (Target Mask)
        # 这是一个下三角矩阵，用于实现自回归（Auto-regressive）特性
        # 确保预测第 t 个词时，只能看到第 1 到 t 的词，看不到 t+1 之后的词
        # shape: (T, T)
        tgt_mask = torch.tril(torch.ones((T, T), device=features.device))

        # 4. Transformer 解码器 (Transformer Decoder)
        # 将文本作为 tgt (Target)，图像作为 memory (Source)
        # decoder_output shape: (N, T, W)
        decoder_output = self.transformer(
            tgt=caption_embed, memory=projected_features, tgt_mask=tgt_mask
        )

        # 5. 输出投影 (Output Projection)
        # 将隐藏层状态投影到词汇表大小，得到每个位置的词汇概率分布 Logits
        # (N, T, W) -> (N, T, V)
        scores = self.output(decoder_output)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output



class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        Inputs:
         - img_size: Size of input image (assumed square).
         - patch_size: Size of each patch (assumed square).
         - in_channels: Number of image channels.
         - embed_dim: Embedding dimension for each patch.
         - num_layers: Number of Transformer encoder layers.
         - num_heads: Number of attention heads.
         - dim_feedforward: Hidden size of feedforward network.
         - num_classes: Number of classification labels.
         - dropout: Dropout probability.
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer to predict class scores from pooled token.
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Inputs:
         - x: Input image tensor of shape (N, C, H, W)

        Returns:
         - logits: Output classification logits of shape (N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # TODO: Implement the forward pass of the Vision Transformer.             #
        # 1. Convert the input image into a sequence of patch vectors.            #
        # 2. Add positional encodings to retain spatial information.              #
        # 3. Pass the sequence through the Transformer encoder.                   #
        # 4. Average pool patch vectors to get a feature vector for each image.   #
        #    You may find torch.mean useful.                                      #
        # 5. Feed it through a linear layer to produce class logits.              #
        ############################################################################
        # 1. 图像分块与嵌入 (Patch Embedding)
        # 将输入图像切分为固定大小的补丁(patches)，并映射为嵌入向量
        # 输入形状: (N, C, H, W)
        # 输出形状: (N, Num_Patches, Embed_Dim)
        x = self.patch_embed(x)

        # 2. 添加位置编码 (Add Positional Encoding)
        # 将位置信息加到嵌入向量上，使模型知道每个补丁在原图中的位置
        # 输出形状保持不变: (N, Num_Patches, Embed_Dim)
        x = self.positional_encoding(x)

        # 3. Transformer 编码器处理 (Transformer Encoder)
        # 通过多层 Transformer Encoder 处理序列，进行自注意力交互
        # 输出形状保持不变: (N, Num_Patches, Embed_Dim)
        x = self.transformer(x)

        # 4. 全局平均池化 (Global Average Pooling)
        # 对所有补丁的特征向量求平均，得到整张图像的特征表示
        # 这一步消除了序列长度维度 (dim=1)
        # 输出形状: (N, Embed_Dim)
        x = torch.mean(x, dim=1)

        # 5. 分类头 (Classification Head)
        # 将图像特征向量通过线性层投影到类别空间，得到分类分数
        # 输出形状: (N, Num_Classes)
        logits = self.head(x)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


        return logits
