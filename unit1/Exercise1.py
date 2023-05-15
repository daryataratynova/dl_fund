#import libraries
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [0 for _ in range(num_features)] #w_[i] = 0 
        self.bias = 0 #b = 0 

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
        error = true_y - prediction

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
        if error_count == 0: break #EXERCISE 1

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


ppn = Perceptron(num_features=2)

train(model=ppn, all_x=X_train, all_y=y_train, epochs=5)

train_acc = compute_accuracy(ppn, X_train, y_train)
print(train_acc)

x1_min, x1_max, x2_min, x2_max = plot_boundary(ppn)

#Visualization 

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