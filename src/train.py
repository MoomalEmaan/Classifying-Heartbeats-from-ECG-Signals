#Task 4 train the network, choose the learning rate and save the best parameters
#Run grad_check.py first because Table 2 uses its result
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import ECGNet, cross_entropy
from prepare import prepare_data

BATCH_SIZE = 128
BETA = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 30
ALPHAS = [0.01, 0.1, 0.5]
HARDWARE = "CPU: Intel Core i5-6300U @ 2.40GHz"


#Loss and accuracy on a whole set of rows, with no gradients needed
def evaluate(model, x, y):
    with torch.no_grad():
        logits = model(x)
        loss = cross_entropy(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


def train(alpha):
    #A fresh network with seed 0 for every learning rate
    torch.manual_seed(0)
    model = ECGNet()

    #One velocity per parameter, starting at zero
    velocity = {}
    for name, p in model.named_parameters():
        velocity[name] = torch.zeros_like(p)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    n = len(X_train)

    for epoch in range(EPOCHS):
        #Shuffle the rows every epoch
        order = torch.randperm(n)

        for start in range(0, n, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            loss = cross_entropy(model(X_train[batch]), Y_train[batch])

            #Clear the old gradients, then compute the new ones
            model.zero_grad()
            loss.backward()

            #The update rule from Task 4
            with torch.no_grad():
                for name, p in model.named_parameters():
                    g = p.grad
                    #Weight decay on the weights only, not the biases
                    if name.startswith("W"):
                        g = g + WEIGHT_DECAY * p
                    velocity[name] = BETA * velocity[name] + (1 - BETA) * g
                    p -= alpha * velocity[name]

        #Record the results after every epoch
        train_loss, train_acc = evaluate(model, X_train, Y_train)
        val_loss, val_acc = evaluate(model, X_val, Y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"alpha {alpha} epoch {epoch + 1}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")

        #Keep a copy of the parameters from the best epoch so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {}
            for name, value in model.state_dict().items():
                best_state[name] = value.clone()

    #Put the best parameters back into the model
    model.load_state_dict(best_state)
    return model, history, best_epoch


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data()

    #Initial loss on the training rows, before any training
    torch.manual_seed(0)
    initial_loss, _ = evaluate(ECGNet(), X_train, Y_train)
    print("Initial loss:", initial_loss)

    #Task 4.2 train once for each learning rate
    results = {}
    for alpha in ALPHAS:
        start_time = time.time()
        model, history, best_epoch = train(alpha)
        results[alpha] = {
            "model": model,
            "history": history,
            "best_epoch": best_epoch,
            "seconds": time.time() - start_time,
        }

    #Table 3: lowest validation loss and the validation accuracy at that epoch
    rows = []
    for alpha in ALPHAS:
        history = results[alpha]["history"]
        epoch_index = results[alpha]["best_epoch"] - 1
        rows.append({
            "alpha": alpha,
            "lowest_val_loss": history["val_loss"][epoch_index],
            "val_accuracy_at_that_epoch": history["val_acc"][epoch_index],
        })
    table3 = pd.DataFrame(rows)
    table3.to_csv("results/tables/table3_learning_rate.csv", index=False)
    print(table3.to_string(index=False))

    #Choose the learning rate with the lowest validation loss
    best_alpha = ALPHAS[table3["lowest_val_loss"].argmin()]
    best = results[best_alpha]
    print("Chosen alpha:", best_alpha)

    #Task 4.3 save the parameters from the best epoch
    torch.save(best["model"].state_dict(), "results/best_model.pt")
    print("Saved results/best_model.pt")

    #Figure 2: validation loss for the three learning rates
    plt.figure(figsize=(7, 4))
    for alpha in ALPHAS:
        plt.plot(range(1, EPOCHS + 1), results[alpha]["history"]["val_loss"], label=f"alpha = {alpha}")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title("Validation loss for each learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/val_loss_by_alpha.png", dpi=150)
    plt.close()

    #Figure 3: training and validation loss for the chosen learning rate
    history = best["history"]
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="training")
    plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss for alpha = {best_alpha}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/loss_curves.png", dpi=150)
    plt.close()

    #Figure 3: training and validation accuracy for the chosen learning rate
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), history["train_acc"], label="training")
    plt.plot(range(1, EPOCHS + 1), history["val_acc"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy for alpha = {best_alpha}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/accuracy_curves.png", dpi=150)
    plt.close()

    #Table 2: network and training
    grad_check = pd.read_csv("results/tables/grad_check.csv")
    epoch_index = best["best_epoch"] - 1
    total_parameters = sum(p.numel() for p in best["model"].parameters())
    table2 = pd.DataFrame({
        "Item": [
            "Trainable parameters",
            "Initial loss (before training)",
            "Gradient check: max difference for W3 / b3",
            "Chosen alpha",
            "Best epoch (lowest validation loss)",
            "Training / validation loss at best epoch",
            "Training / validation accuracy at best epoch",
            "Training time and hardware",
        ],
        "Value": [
            total_parameters,
            f"{initial_loss:.4f}",
            f"{grad_check['diff_W3'][0]:.2e} / {grad_check['diff_b3'][0]:.2e}",
            best_alpha,
            best["best_epoch"],
            f"{history['train_loss'][epoch_index]:.4f} / {history['val_loss'][epoch_index]:.4f}",
            f"{history['train_acc'][epoch_index]:.4f} / {history['val_acc'][epoch_index]:.4f}",
            f"{best['seconds']:.0f} seconds, {HARDWARE}",
        ],
    })
    table2.to_csv("results/tables/table2_network_training.csv", index=False)
    print(table2.to_string(index=False))
    print("Saved the tables and figures in results/")
