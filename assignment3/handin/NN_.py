
import numpy as np
import random

class Neuron_Network():

    def __init__(self, nr_of_inputs, nr_of_hiddens, nr_of_outputs):

        self.inputs = nr_of_inputs
        self.hiddens = nr_of_hiddens
        self.outputs = nr_of_outputs
        self.W1 = np.random.randn(self.inputs, self.hiddens)
        self.W2 = np.random.randn(self.hiddens, self.outputs)
        self.B1 = [random.random() for _ in range(0, self.hiddens)]
        self.B2 = [random.random() for _ in range(0, self.outputs)]

    def transfer(self, a):
        return a / (1 + np.abs(a))

    def transfer_prime(self, a):
        return 1 / ((1 + np.abs(a)) ** 2)

    def sum_of_squares(self, xs, tks):
        yks = self.forward(xs)
        return 0.5 * (sum(yks - tks) ** 2)

    def forward(self, xs):
        self.am = np.dot(xs, self.W1)
        # add hidden layer bias
        rows, _ = xs.shape
        self.hidden_bias = self.make_bias(rows)
        self.am += self.hidden_bias

        # pass hidden layer through activation/transfer function
        self.zm = self.transfer(self.am)

        ak = np.dot(self.zm, self.W2)
        # add output bias
        ak += self.B2
        return ak

    def backpropagation(self, xs, tks):
        yks = self.forward(xs)

        delta_k = yks - tks
        dEdW2 = np.dot(self.zm.T, delta_k)

        # Nx1 `dot` 1x3 => Nx3 `mult` Nx3 => Nx3
        delta_j = np.dot(delta_k, self.W2.T) * self.transfer_prime(self.am)
        # 1xN `dot` Nx3 => 1x3
        dEdW1 = np.dot(xs.T, delta_j)

        return dEdW1, dEdW2

    # Helper to create bias array for hidden layer
    def make_bias(self, rows):
        cols = []
        for bias in self.B1:
            col = np.empty((rows, 1))
            col.fill(bias)
            cols.append(col)
        return np.concatenate(cols, axis=1)


    def train(self, xs, ys, epochs):
        current_cost = self.sum_of_squares(xs, ys)
        original_cost = current_cost
        scale = 2
        initial_step = 2
        iterations = epochs
        costs = []
        # train
        for _ in range(iterations):
            preW1 = np.array(self.W1)
            preW2 = np.array(self.W2)

            dEdW1, dEdW2 = self.backpropagation(xs, ys)

            # add to weights
            W1add = (preW1 + dEdW1) * scale
            W2add = (preW2 + dEdW2) * scale
            # subtract from weights
            W1sub = (preW1 - dEdW1) * scale
            W2sub = (preW2 - dEdW2) * scale

            # compare costs
            self.W1 = W1add
            self.W2 = W2add
            costAdd = self.sum_of_squares(xs, ys)
            self.W1 = W1sub
            self.W2 = W2sub
            costSub = self.sum_of_squares(xs, ys)

            # print "The cost to beat: " + str(current_cost[0])

            if costAdd <= current_cost:
                # Set the adding-weights as current weights
                # print 'Cost reduced by adding gradients'
                self.W1 = W1add
                self.W2 = W2add
                current_cost = costAdd
                costs.append(current_cost)
            elif costSub <= current_cost:
                # Set the subtracting-weights as current weights
                self.W1 = W1sub
                self.W2 = W2sub
                current_cost = costSub
                costs.append(current_cost)
            else:
                self.W1 = preW1
                self.W2 = preW2
                # Keep gradients as is but take smaller steps, current_cost
                # is still the cost to beat
                scale /= 10
                if scale < initial_step/10:
                    scale = initial_step

        print \
            'Using {0} hidden neurons, the cost is reduced by {1:.3f}% from {2:.3f} to {3:.3f}'\
                .format(self.hiddens,
                        (1-current_cost[0]/original_cost[0])*100,
                        original_cost[0],
                        current_cost[0])
        return current_cost, costs






