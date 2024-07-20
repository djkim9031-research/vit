import math
import torch
from torch import nn


class GELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space
    """

    def __init__(self, config):
        super().__init__()
        self.width = config["image_width"]
        self.height = config["image_height"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.width // self.patch_size) * (self.height // self.patch_size)
        # output shape = (hidden_size, height//patch_size, width//patch_size)
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, 
                                    kernel_size = self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        # B, C, H, W => B, hidden, H/num_patches, W/num_patches
        x = self.projection(x)
        # => B, num_patches, hidden
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token, which is added to the beginning of 
        # the input sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1) #(batch_size, num_patches + 1, hidden_size)
        x = x + self.position_embedding
        return x
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        # Projection of embedding tensors to qkv space
        self.qkv_bias = config["qkv_bias"]
        self.qkv_projection = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=self.qkv_bias)
        
        # Projection of the calculated attention tensor
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, output_attentions=False):
        qkv = self.qkv_projection(x) # (batch_size, num_patches + 1, 3*hidden_size)

        # split qkv into each q, k, v
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, seq_len, _ = query.size()
        # each q, k, v will be of size (batch_size, num_attention_heads, seq_len, head_size)
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Calculate the weight (= Softmax(Q*K.T/sqrt(head_size)))
        weights = torch.matmul(query, key.transpose(-1, -2)) # (batch_size, num_heads, seq_len, seq_len)
        weights = weights / math.sqrt(self.attention_head_size)
        attn_probs = nn.functional.softmax(weights, dim=-1)
        
        # Calculate the attention output
        attn_out = torch.matmul(attn_probs, value) #(batch_size, num_heads, seq_len, head_size)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Projection
        attn_out = self.output_projection(attn_out) #(batch_size, seq_len, hidden_size)

        if not output_attentions:
            return(attn_out, None)
        else:
            return(attn_out, attn_probs)
        
class MLP(nn.Module):
    """
    Multi-layer perceptron in the transformer architecture
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        
    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x
    
class Block(nn.Module):
    """
    A single attention block
    """

    def __init__(self, config):
        super().__init__()
        # normalize over the last input dimension (=hidden_size)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.attention = MultiHeadAttention(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)

    def forward(self, x, output_attentions=False):
        # self-attention
        attn_out, attn_probs = self.attention(self.layernorm_1(x), output_attentions)

        # residual connection
        x = x + attn_out

        # feed-forward
        mlp_out = self.mlp(self.layernorm_2(x))

        # residual connection
        x = x + mlp_out

        if not output_attentions:
            return (x, None)
        else:
            return (x, attn_probs)
        
class Encoder(nn.Module):
    """
    The encoder module of the transformer
    """

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attn_probs = block(x, output_attentions)
            if output_attentions:
                all_attentions.append(attn_probs)

        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        
class ViT(nn.Module):
    """
    The ViT model for image classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_width = config["image_width"]
        self.image_height = config["image_height"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        # Construct transformer building blocks
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embd_out = self.embeddings(x)
        attn_out, all_attns = self.encoder(embd_out, output_attentions)
        # The first element in sequence (corrs to cls_token) is responsible for 
        # learning features classification 
        logits = self.classifier(attn_out[:, 0, :])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attns)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embedding.data = nn.init.trunc_normal_(
                module.position_embedding.data.to(torch.float32),
                mean = 0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embedding.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean = 0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
     





