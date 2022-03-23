# -*- coding: utf-8 -*-
import json
import sys
import numpy as np


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        # dC/dz
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # dC/dz
        return (a-y)


class Network(object):
    def __init__(self, sizes:list, cost=CrossEntropyCost) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a:np.ndarray) -> np.ndarray:
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w@a+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_count, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data is not None:
            n_data = len(evaluation_data)
        n = len(training_data)
        mini_batch_size = int(n/mini_batch_count)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size, :] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data}")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for xy in mini_batch:
            x = xy[:self.sizes[0]].reshape(-1, 1)
            y = xy[self.sizes[0]:].reshape(-1, 1)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = w@activation+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = delta@activations[-2].T
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = self.weights[-l+1].T@delta * sp
            nabla_b[-l] = delta
            nabla_w[-l] = (delta.reshape(-1, 1))@(activations[-l-1].reshape(1, -1))
        return (nabla_b, nabla_w)

    def accuracy(self, data):
        acc_count = 0
        for xy in data:
            x = xy[:self.sizes[0]].reshape(-1, 1)
            y = xy[self.sizes[0]:].reshape(-1, 1)
            acc_count += int(np.argmax(self.feedforward(x)) == np.argmax(y))
        return acc_count

    def total_cost(self, data, lmbda):
        cost = 0.0
        for xy in data:
            x = xy[:self.sizes[0]].reshape(-1, 1)
            y = xy[self.sizes[0]:].reshape(-1, 1)
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    # Loading a Network
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
    with np.load("train_data.npz") as f:
        train_images = f['train_images']
        train_labels = f['train_lables'].astype(np.float32)
        train_images = train_images.reshape(train_images.shape[0], -1)
        train_data = np.c_[train_images, train_labels]
    with np.load("test_data.npz") as f:
        test_images = f['test_images']
        test_labels = f['test_lables'].astype(np.float32)
        test_images = test_images.reshape(test_images.shape[0], -1)
        test_data = np.c_[test_images, test_labels]
    nn = Network([784, 15, 10])
    nn.SGD(train_data, 50, 10, 0.07, 0, monitor_training_accuracy=True)
    nn.save("nn.json")
