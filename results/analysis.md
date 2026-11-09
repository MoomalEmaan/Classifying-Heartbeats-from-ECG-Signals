# Task 6 - Analysis

The results below come from the saved model (learning rate 0.5, epoch 23), tested once on the 21,892 test rows.

## Task 6.1

**Question:** A model that always answers "N" would get what test accuracy? Compute it from the class table. Compare it with your model's accuracy. What does this tell you about using accuracy alone?

**Answer:**

A model that always answers "N" gets 18,118 of the 21,892 test rows right, so its accuracy is 18,118 / 21,892 = **82.76%**. Our network reaches **97.82%**.

Accuracy alone does not tell the full story. The always-"N" model scores nearly 83% without finding a single abnormal beat, because most of the beats are normal. Its balanced accuracy would be only 20%, because it gets 100% on N and 0% on the other four classes.

The same effect is hidden inside our own result. Test accuracy is 97.82%, but balanced accuracy is 83.65%, and the network recognises only 66.37% of S beats and 62.96% of F beats. Accuracy is dominated by class N, so per-class and balanced accuracy should be reported with it, especially because the rare beats can be the ones that matter most.

## Task 6.2

**Question:** Which class has the lowest per-class accuracy? Using the confusion matrix, which class is it most often mistaken for, and why might those two beat types be hard to separate?

**Answer:**

**F (fusion)** has the lowest per-class accuracy, at 62.96% (102 of 162 beats correct). It is most often mistaken for **N**: 39 F beats (24.1%) were predicted as N. The next most common mistake was V, with 21 beats (13.0%).

These two types may be hard to separate because a fusion beat happens when a normal beat and a ventricular beat occur together, so it can share parts of the normal beat's shape. This also fits the second most common mistake, V. Class size probably makes it harder too: the training file has 641 F rows and 72,471 N rows, so N has far more influence on the loss during training.

## Task 6.3

**Question:** Look at your loss curves. Is there a gap between training and validation loss? Is the network overfitting, underfitting, or neither? Explain from the curves.

**Answer:**

There is a small gap between the training and validation loss. The two curves start almost together, but validation loss stays above training loss in later epochs. At the best epoch (23), training loss is 0.0560 and validation loss is 0.0722. Training and validation accuracy are 98.26% and 97.93%, a gap of only 0.33 percentage points.

The network shows **mild overfitting**. Training loss keeps falling, but validation loss has almost stopped improving after about epoch 20 (0.0759 at epoch 20, 0.0722 at epoch 23 and 0.0750 at epoch 30). It is not underfitting, because both losses are low and both accuracies are high. The overfitting is not severe, because validation loss does not rise sharply. Saving the parameters from the epoch with the lowest validation loss also protects the final model from later overfitting.

## Task 6.4

**Question:** Why did you not initialise every weight to zero? Why does the He rule use √(2/n[ℓ-1]) for ReLU layers? (Lecture notes, Section 20.2.)

**Answer:**

If every weight and bias starts at zero, all the logits are zero, so the network gives the same output for every heartbeat. We tested this on our network: the gradients of all the weights and of the hidden layer biases were exactly zero, and only the output bias `b3` received a gradient. The network could not learn to use the input at all. More generally, starting all the weights with the same value creates symmetry: units in a layer compute the same activation, receive the same gradient and learn the same thing. Random starting weights break this symmetry.

The He rule keeps the size of the signal steady from layer to layer. A unit adds up its n[ℓ-1] weighted inputs, so the variance of its output grows with the number of inputs and with the variance of its weights. ReLU sets negative values to zero, which removes about half of the signal's variance. To balance both effects, the weights use a variance of 2/n[ℓ-1], which is a standard deviation of √(2/n[ℓ-1]). The n cancels the effect of adding many inputs, and the 2 makes up for the half removed by ReLU. Without this scaling, the signal could shrink towards zero or grow very large as it passes through the layers.

## Task 6.5

**Question:** What does momentum do to the update when two consecutive mini-batch gradients point in opposite directions? What does weight decay do to a weight that receives no gradient?

**Answer:**

**Momentum:** the update follows the velocity, v = βv + (1 - β)g, not just the latest gradient. When two consecutive gradients point in opposite directions, they largely cancel, so the update becomes small and the weights do not jump back and forth. For example, with β = 0.9 and starting from v = 0, a gradient of +1 gives v = 0.1, and then a gradient of -1 gives v = -0.01. This is how momentum smooths out noisy mini-batch gradients.

**Weight decay:** if a weight receives no gradient from the loss, then ∂J/∂P = 0, so g = λP. The weight is still updated, and it shrinks slowly towards zero. With α = 0.5 and λ = 0.0001, once the velocity has settled, the weight is multiplied by about 1 - αλ = 0.99995 on each step. Over the 616 steps in one epoch, that is a shrinkage of about 3%. This keeps weights small unless the data keeps giving them a reason to be large.

## Task 6.6

**Question:** The F class has only 641 training rows. Suggest one change to the data or the loss that could help the network on rare classes. Explain the idea; do not run it.

**Answer:**

One change is to use a **class-weighted cross-entropy loss**. Each row's loss would be multiplied by a weight that is larger for rare classes, for example:

weight of class k = (number of training rows) / (5 × number of training rows in class k)

Using the training file's counts, F would get a weight of about 27.3 and N about 0.24. A mistake on an F beat would therefore cost about 113 times as much as a mistake on an N beat, so the network could no longer ignore the rare classes to get a low average loss.

The weights must be calculated from the training rows only, never the test rows. The likely trade-off is more false alarms, with some N beats predicted as rare classes, so overall accuracy could fall even if balanced accuracy improves. This idea was not run; it is only a suggestion.
