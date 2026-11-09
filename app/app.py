#Task 7.1 front end. Run from the project folder with: streamlit run app/app.py
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

sys.path.append("src")
from model import ECGNet

CLASS_NAMES = ["N: normal", "S: supraventricular ectopic", "V: ventricular ectopic",
               "F: fusion", "Q: unknown / paced"]
FS = 125  # sampling frequency in Hz


#Load these only once, not every time the page updates
@st.cache_resource
def load_model():
    model = ECGNet()
    model.load_state_dict(torch.load("results/best_model.pt"))
    return model


@st.cache_data
def load_test_rows():
    return pd.read_csv("data/mitbih_test.csv", header=None)


#Pick a random test row, of one class or of any class
def pick_random_row():
    test_rows = load_test_rows()
    choice = st.session_state["class_choice"]
    if choice != "Any class":
        test_rows = test_rows[test_rows[187] == CLASS_NAMES.index(choice)]
    st.session_state["row"] = int(test_rows.sample(1).index[0])


#Use the full width of the screen so everything fits without scrolling
st.set_page_config(page_title="ECG heartbeat classifier", layout="wide")

#The row number starts at 0
if "row" not in st.session_state:
    st.session_state["row"] = 0

st.title("Classifying heartbeats from ECG signals")
st.caption("A small neural network built from scratch in PyTorch. Course project, not a medical tool.")

signal = None
true_label = None
true_class = "Not known"

#The controls go in the sidebar
with st.sidebar:
    st.header("Choose a heartbeat")
    source = st.radio("Source", ["Test-set row number", "Upload a CSV file"])

    if source == "Test-set row number":
        test_rows = load_test_rows()

        #Random beat button
        st.selectbox("Class for a random beat", ["Any class"] + CLASS_NAMES, key="class_choice")
        st.button("Show a random beat", on_click=pick_random_row)

        row_number = st.number_input("Row number", min_value=0, max_value=len(test_rows) - 1, key="row")
        signal = test_rows.iloc[row_number, :187].to_numpy(dtype=float)
        true_label = int(test_rows.iloc[row_number, 187])
        true_class = CLASS_NAMES[true_label]
    else:
        uploaded_file = st.file_uploader("CSV file with one row of 187 values", type="csv")
        if uploaded_file is not None:
            uploaded_rows = pd.read_csv(uploaded_file, header=None)
            if uploaded_rows.shape == (1, 187):
                signal = uploaded_rows.iloc[0].to_numpy(dtype=float)
            else:
                st.error(f"The file must have 1 row and 187 values. This file has shape {uploaded_rows.shape}.")

if signal is None:
    st.info("Upload a CSV file with one row of 187 values to see the prediction.")
else:
    #Predict the class and the probability of all five classes
    x = torch.tensor(signal, dtype=torch.float32).reshape(1, 187)
    with torch.no_grad():
        probabilities = torch.softmax(load_model()(x), dim=1)[0].tolist()
    predicted = probabilities.index(max(probabilities))

    #Show the main results in one row
    col1, col2, col3 = st.columns(3)
    col1.caption("Predicted class")
    col1.subheader(CLASS_NAMES[predicted])
    col2.caption("Probability of predicted class")
    col2.subheader(f"{probabilities[predicted]:.1%}")
    col3.caption("True class")
    col3.subheader(true_class)

    #Correct or wrong badge, only when the true class is known
    if true_label is not None:
        if predicted == true_label:
            st.success("Correct: the network predicted the true class.")
        else:
            st.error(f"Wrong: the network predicted {CLASS_NAMES[predicted]}, but the true class is {true_class}.")

    #Show the two charts side by side
    left, right = st.columns(2)

    #Plot the heartbeat
    time_ms = [1000 * i / FS for i in range(187)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(time_ms, signal)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalised amplitude")
    ax.set_title("Heartbeat")
    fig.tight_layout()
    left.pyplot(fig)

    #Bar chart of the probabilities
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(CLASS_NAMES, probabilities)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("probability")
    ax.set_title("Probability of each class")
    fig.tight_layout()
    right.pyplot(fig)
