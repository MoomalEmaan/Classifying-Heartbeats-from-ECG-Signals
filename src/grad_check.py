#Task 3.2 compare my hand-computed output layer gradients with PyTorch's
import pandas as pd
import torch

from model import ECGNet, cross_entropy
from prepare import prepare_data

X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()

#Take the first 64 training rows
B = 64
x = X_train[:B]
y = Y_train[:B]

#Forward pass and backward pass on a fresh network
torch.manual_seed(0)
model = ECGNet()
logits = model(x)
loss = cross_entropy(logits, y)
loss.backward()
print("Loss on the first 64 rows:", loss.item())

with torch.no_grad():
    #A2: activations of the second hidden layer
    a1 = torch.clamp(x @ model.W1.T + model.b1, min=0)
    a2 = torch.clamp(a1 @ model.W2.T + model.b2, min=0)

    #Y_hat: softmax probabilities, B x 5
    Y_hat = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))

    #Y: one-hot labels, B x 5
    Y = torch.zeros(B, 5)
    Y[torch.arange(B), y] = 1

    #The two formulas from Task 3.2
    grad_W3 = (Y_hat - Y).T @ a2 / B
    grad_b3 = (Y_hat - Y).sum(dim=0) / B

    #Largest absolute difference from the gradients PyTorch stored
    diff_W3 = (grad_W3 - model.W3.grad).abs().max().item()
    diff_b3 = (grad_b3 - model.b3.grad).abs().max().item()

print("Max difference for W3:", diff_W3)
print("Max difference for b3:", diff_b3)
print("Both within 1e-5:", diff_W3 < 1e-5 and diff_b3 < 1e-5)

#Saving the result for table 2
pd.DataFrame({"diff_W3": [diff_W3], "diff_b3": [diff_b3]}).to_csv("results/tables/grad_check.csv", index=False)
print("Saved results/tables/grad_check.csv")
