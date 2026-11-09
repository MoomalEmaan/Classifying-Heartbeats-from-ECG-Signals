#Task 1 load and prepare the data 
import pandas as pd
from load_data import load_raw
import numpy as np
import torch

def prepare_data():
    #loading the training and testing data
    raw_data=load_raw()
    train_data=raw_data['train']
    test_data=raw_data['test']

    #Checking the shape of the data
    print("Training data shape: ",train_data.shape)
    print("Testing data shape: ",test_data.shape)

    # Separating the signal (columns 0–186) from the label (column 187)
    #Training data
    X_train_file=train_data.iloc[:,0:187]
    Y_train_file=train_data.iloc[:, 187]
    #Testing data
    X_test=test_data.iloc[:,0:187]
    Y_test=test_data.iloc[:, 187]

    #Checking the data type and values of label column (Train and test data both)
    print("Data type of training labels before conversion: ",Y_train_file.dtype)
    print("Data type of testing labels before conversion: ",Y_test.dtype)
    print("Training labels before conversion: ",Y_train_file.unique())
    print("Testing labels before conversion: ",Y_test.unique())

    #Converting the training and testing columns data type from float to int 
    Y_train_file=Y_train_file.astype(int)
    Y_test=Y_test.astype(int)

    #Checking the values and data type of labels after conveting the data type
    print("Data type of training labels after conversion: ",Y_train_file.dtype)
    print("Data type of testing labels after conversion: ",Y_test.dtype)
    print("Training labels after conversion: ",Y_train_file.unique())
    print("Testing labels after conversion: ",Y_test.unique())

    #Checking the signal shape 
    print("Signal shape of training data: ",X_train_file.shape)
    print("Signal shape of testing data: ",X_test.shape)

    #Finding Minimum and maximum signal value 
    signal_min = X_train_file.min().min()
    signal_max = X_train_file.max().max()
    print("Minimum value of signal: ",signal_min)
    print("Maximum value of signal: ",signal_max)

    #checking no of rows per class in both files 
    train_counts = Y_train_file.value_counts().sort_index()
    test_counts = Y_test.value_counts().sort_index()
    print("Train file labels and counts:",train_counts)
    print("Test file labels and counts:",test_counts)

    #split the training file into training and validation data 
    #Produce row numbers for validation and training set
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_data))
    n_val = int(0.1 * len(train_data))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    #Assign rows to validation and training data
    X_val=X_train_file.iloc[val_idx]
    Y_val=Y_train_file.iloc[val_idx]
    X_train=X_train_file.iloc[train_idx]
    Y_train=Y_train_file.iloc[train_idx]

    #Number of rows in train,validation and test data 
    print("Number of rows in train data: ",len(X_train))
    print("Number of rows in validation data: ",len(X_val))
    print("Number of rows in test data: ",len(X_test))

    #Convert the signals to torch.float32 tensors and the labels to torch.long tensors
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.to_numpy(), dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.to_numpy(), dtype=torch.long)

    #Check the shape and data type of each tensor
    print("Train signals:", X_train_tensor.shape, X_train_tensor.dtype)
    print("Train labels:", Y_train_tensor.shape, Y_train_tensor.dtype)
    print("Validation signals:", X_val_tensor.shape, X_val_tensor.dtype)
    print("Validation labels:", Y_val_tensor.shape, Y_val_tensor.dtype)
    print("Test signals:", X_test_tensor.shape, X_test_tensor.dtype)
    print("Test labels:", Y_test_tensor.shape, Y_test_tensor.dtype)

    #create Table 1 and save it in the results folder
    table1 = pd.DataFrame({
        "Item": [
            "Train / validation / test rows",
            "Minimum / maximum signal value",
            "Rows per class, train file (N, S, V, F, Q)",
            "Rows per class, test file (N, S, V, F, Q)",
        ],
        "Value": [
            f"{len(X_train)} / {len(X_val)} / {len(X_test)}",
            f"{signal_min} / {signal_max}",
            " / ".join(map(str, train_counts)),
            " / ".join(map(str, test_counts)),
        ],
    })
    table1.to_csv("results/tables/table1_data.csv", index=False)
    print(table1.to_string(index=False))
    print("Saved results/tables/table1_data.csv")

    return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor


if __name__ == "__main__":
    prepare_data()
