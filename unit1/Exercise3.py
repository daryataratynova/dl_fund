#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

#EXE 3.3: No changes, the same as before introducting alpha to original code
#EXE 3.4: With learning rate = 0.5 less erros but still the same number of epochs required. 
#epoch 1 errors = 0.5, epoch 2 errors = 1.5, epoch 3 errors = 0.5, then errors = 0
#EXE 3.5 With small random weights training with learning rate = 0.5 requires 3 epochs for errors = 0 (errors =1 for first 2 epochs), while
# with learning rate = 1.0 requires also 3 epochs for errors = 0 but epoch 1 errors = 1, epoch 2 errors = 1

class Perceptron:
    def __init__(self, num_features, alpha):
        random.seed(123)
        self.num_features = num_features
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_features)] #w_[i] = 0 
        self.bias = random.uniform(-0.5, 0.5) #b = 0 
        self.alpha = alpha

    def forward(self, x):
        weighted_sum_z = self.bias #z = b
        #update z using weights and each feature
        for i, _ in enumerate(self.weights):
            weighted_sum_z += x[i] * self.weights[i]
        
        #obtain class label
        if weighted_sum_z > 0.0:
            prediction = 1
        else:
            prediction = 0

        return prediction
    
    def update(self , x, true_y):
        prediction = self.forward(x)
        error = self.alpha* (true_y - prediction)
        # update bias and weights iterating thr. each  feature 
        self.bias += error
        for i, _ in enumerate(self.weights):
            self.weights[i] += error * x[i]

        return error
    
def train(model, all_x, all_y, epochs):

    for epoch in range(epochs):
        error_count = 0

        for x, y in zip(all_x, all_y):
            error = model.update(x, y)
            error_count += abs(error)
        print(f"Epoch {epoch+1} errors {error_count}")
        

def compute_accuracy(model, all_x, all_y):

    correct = 0.0

    for x, y in zip(all_x, all_y):
        prediction = model.forward(x)
        correct += int(prediction == y)

    return correct / len(all_y)

def plot_boundary(model):

    w1, w2 = model.weights[0], model.weights[1]
    b = model.bias

    x1_min = -20
    x2_min = (-(w1 * x1_min) - b) / w2

    x1_max = 20
    x2_max = (-(w1 * x1_max) - b) / w2

    return x1_min, x1_max, x2_min, x2_max

#import dataset that includes x1 x2 and label y
df = pd.read_csv("unit1/perceptron_toydata-truncated.txt", sep="\t")

#features x1 x2 go to X and labels to Y
X_train = df[["x1", "x2"]].values
y_train = df["label"].values


ppn = Perceptron(num_features=2, alpha = 0.5)

train(model=ppn, all_x=X_train, all_y=y_train, epochs=5)

train_acc = compute_accuracy(ppn, X_train, y_train)
print(train_acc)

x1_min, x1_max, x2_min, x2_max = plot_boundary(ppn)

#Vizualization 

#Class 0
plt.plot(
    X_train[y_train == 0, 0], #x1 feature
    X_train[y_train == 0, 1], #x2 feature
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

#Class 1
plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()