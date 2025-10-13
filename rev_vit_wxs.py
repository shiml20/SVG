import torch
from torch import nn
# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA


class RevViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        enable_amp=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(
            [
                ReversibleBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x):

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        for layer in self.layers:
            X_1, X_2 = layer(X_1, X_2)

        return torch.cat([X_1, X_2], dim=-1)
    
    def rev_forward(self, y):
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        for layer in self.layers[::-1]:
            Y_1, Y_2 = layer.rev_forward(Y_1, Y_2)
        return torch.cat([Y_1, Y_2], dim=-1)

class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, dim, num_heads, enable_amp):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, enable_amp=enable_amp
        )

        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2

        # free memory since X_1 is now not needed
        del X_1

        g_Y_1 = self.G(Y_1)

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2
    
    def rev_forward(self, Y_1, Y_2):
        
        g_Y_1 = self.G(Y_1)
        X_2 = Y_2 - g_Y_1
        f_X_2 = self.F(X_2)
        X_1 = Y_1 - f_X_2
        return X_1, X_2

class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = MHA(dim, num_heads, batch_first=True)
        self.enable_amp = enable_amp

    def forward(self, x):
        # See MLP fwd pass for explanation.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            out, _ = self.attn(x, x, x)
            return out


def main():
    """
    This is a simple test to check if the recomputation is correct
    by computing gradients of the first learnable parameters twice --
    once with the custom backward and once with the vanilla backward.

    The difference should be ~zero.
    """

    # insitantiating and fixing the model.
    model = RevViT(embed_dim=192)

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.rand((1, 3, 384))

    # output of the model under reversible backward logic
    output = model(x)

    x_recon = model.rev_forward(output)
    print(x.shape)
    print(output.shape)
    assert torch.allclose(x, x_recon)


if __name__ == "__main__":
    main()