#Task 5.2 the three metric functions
import torch


#Share of rows where the prediction equals the true label
def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()


#C[j, k] = number of rows whose true class is j and predicted class is k
def confusion_matrix(y_true, y_pred):
    C = torch.zeros(5, 5, dtype=torch.long)
    for true_class, predicted_class in zip(y_true.tolist(), y_pred.tolist()):
        C[true_class, predicted_class] += 1
    return C


#For each class: correct rows divided by all rows of that class
def per_class_accuracy(C):
    return (C.diag() / C.sum(dim=1)).tolist()
