#Task 5 test the saved model once on the test rows
import matplotlib.pyplot as plt
import pandas as pd
import torch

from metrics import accuracy, confusion_matrix, per_class_accuracy
from model import ECGNet
from prepare import prepare_data

CLASS_LETTERS = ["N", "S", "V", "F", "Q"]

X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()

#Load the saved parameters
model = ECGNet()
model.load_state_dict(torch.load("results/best_model.pt"))

#The predicted class is the one with the largest logit
with torch.no_grad():
    Y_pred = model(X_test).argmax(dim=1)

#Metrics
test_accuracy = accuracy(Y_test, Y_pred)
C = confusion_matrix(Y_test, Y_pred)
class_accuracy = per_class_accuracy(C)
balanced_accuracy = sum(class_accuracy) / 5
always_n_accuracy = (Y_test == 0).float().mean().item()

print("Test accuracy:", test_accuracy)
print("Balanced accuracy:", balanced_accuracy)
print("Always-N accuracy:", always_n_accuracy)
print("Per-class accuracy:", dict(zip(CLASS_LETTERS, class_accuracy)))
print("Confusion matrix (rows are true classes, columns are predictions):")
print(C)

#Correctness checks
print("Confusion matrix total:", C.sum().item(), "(must be 21892)")
print("Rows add up to:", C.sum(dim=1).tolist(), "(must be [18118, 556, 1448, 162, 1608])")

#Table 4
table4 = pd.DataFrame({
    "Item": ["Per-class N", "Per-class S", "Per-class V", "Per-class F", "Per-class Q",
             "Test accuracy", "Balanced accuracy", "Always-N accuracy"],
    "Value": class_accuracy + [test_accuracy, balanced_accuracy, always_n_accuracy],
})
table4.to_csv("results/tables/table4_test_results.csv", index=False)

#Full confusion matrix with class names on both axes
matrix = pd.DataFrame(C.numpy(), index=CLASS_LETTERS, columns=CLASS_LETTERS)
matrix.to_csv("results/tables/confusion_matrix.csv")

#Figure 4: confusion matrix heat-map, coloured by the share of each true class
shares = C / C.sum(dim=1, keepdim=True)
plt.figure(figsize=(6, 5))
plt.imshow(shares, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="share of the true class")
plt.xticks(range(5), CLASS_LETTERS)
plt.yticks(range(5), CLASS_LETTERS)
plt.xlabel("predicted class")
plt.ylabel("true class")
plt.title("Test confusion matrix (number of beats)")
for j in range(5):
    for k in range(5):
        text_color = "white" if shares[j, k] > 0.5 else "black"
        plt.text(k, j, C[j, k].item(), ha="center", va="center", color=text_color)
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix.png", dpi=150)
plt.close()
print("Saved the table, matrix and heat-map in results/")
