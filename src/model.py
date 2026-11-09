import torch
import torch.nn as nn

#Class ECGNet(nn.Module)that creates every W[ℓ] and b[ℓ] as a nn.Parameter
class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()

        #Layer 1
        self.W1 = nn.Parameter(torch.randn(128, 187)*(2/187)**0.5)
        self.b1 = nn.Parameter(torch.zeros(128))
        #Layer 2
        self.W2 = nn.Parameter(torch.randn(64, 128)*(2/128)**0.5)
        self.b2 = nn.Parameter(torch.zeros(64))
        #Layer 3
        self.W3 = nn.Parameter(torch.randn(5, 64)*(2/64)**0.5)
        self.b3 = nn.Parameter(torch.zeros(5))

    #Forward pass: x has shape (B, 187), the output has shape (B, 5)
    def forward(self, x):
        #Layer 1 with ReLU
        a1 = torch.clamp(x @ self.W1.T + self.b1, min=0)
        #Layer 2 with ReLU
        a2 = torch.clamp(a1 @ self.W2.T + self.b2, min=0)
        #Layer 3 gives the logits, with no activation
        z3 = a2 @ self.W3.T + self.b3
        return z3


#Mean softmax cross-entropy over the batch: L = -z_y + log(sum of e^z)
def cross_entropy(logits, y):
    #Pick the logit of the true class for every row
    z_y = logits[torch.arange(len(y)), y]
    losses = -z_y + torch.logsumexp(logits, dim=1)
    return losses.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ECGNet()
    for name, p in model.named_parameters():
        print(name, p.shape)

    #Total number of trainable parameters
    total = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters:", total)

    #Check the output shape with a batch of 2 rows
    print("Output shape:", model(torch.zeros(2, 187)).shape)