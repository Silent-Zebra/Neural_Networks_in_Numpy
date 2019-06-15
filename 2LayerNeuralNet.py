import csv
import numpy as np


class TwoLayerNN:
    def __init__(self, neurons_hidden: int = 100, neurons_output: int = 10,
                 input_size: int = 784, learn_rate: float = 0.1):
        self.neurons_output = neurons_output
        self.neurons_hidden = neurons_hidden
        self.input_size = input_size

        self.weights_hidden = np.random.rand(input_size, neurons_hidden) / np.math.sqrt(input_size)

        self.weights_output = np.random.rand(neurons_hidden, neurons_output) / np.math.sqrt(input_size)

        self.learn_rate = learn_rate

    def binarize(self, input_vector: np.array):
        new_vector = input_vector > 0
        return new_vector

    def _sigmoid(self, vector: np.array):
        new_vector = 1 / (1 + np.exp(- vector))
        return new_vector

    def _hidden_output_vector(self, input_vector: np.array):
        return self._sigmoid(input_vector @ self.weights_hidden)

    def _prediction_vector(self, input_vector: np.array):
        # 10x1 list output
        values = self._hidden_output_vector(input_vector)
        values = self._sigmoid(values @ self.weights_output)
        return values

    def predict(self, input_vector: np.array):
        vector = self._prediction_vector(input_vector)
        return np.argmax(vector)

    def _generate_label_vector(self, label):
        out = np.zeros((self.neurons_output))
        out[label] = 1
        return out

    def gd(self, batch_labels, batch_data):
        hidden_accumulated_errors = np.zeros((self.input_size, self.neurons_hidden))
        output_accumulated_errors = np.zeros((self.neurons_hidden, self.neurons_output))

        for i in range(len(batch_labels)):
            input_vector = batch_data[i]
            label = batch_labels[i]

            hidden_output_vector = self._hidden_output_vector(input_vector)
            vector = self._prediction_vector(input_vector)
            target = self._generate_label_vector(label)

            output_accumulated_errors = np.zeros(
                (self.neurons_hidden, self.neurons_output))

            output_errors = vector * (1- vector) * (target - vector)

            # @ is matrix multiplication
            sums_to_use = self.weights_output @ output_errors
            deltas = hidden_output_vector * (1 - hidden_output_vector) * sums_to_use

            hidden_accumulated_errors += np.outer(input_vector, deltas)

            temp_prod = np.outer(hidden_output_vector, output_errors)
            output_accumulated_errors += temp_prod

        self.weights_hidden += self.learn_rate * hidden_accumulated_errors

        # Important: output weights have to be updated last
        self.weights_output += self.learn_rate * output_accumulated_errors


agent = TwoLayerNN()

batch_size = 10

epochs = 10

full_data = []

with open('train_5000.csv', 'r') as csvFile:

    reader = csv.reader(csvFile)
    for row in reader:
        try:
            label = int(row[0])
        except:
            continue

        try:
            data = list(map(int, row[1:]))
        except:
            continue

        data = np.array(data)

        data = agent.binarize(data)

        full_data.append((data, label))

csvFile.close()


for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    episode = 0
    batch_data = []
    batch_labels = []

    np.random.shuffle(full_data)

    for i in range(len(full_data)):

        batch_data.append(full_data[i][0])

        batch_labels.append(full_data[i][1])

        if i % batch_size == batch_size-1:
            agent.gd(batch_labels, batch_data)
            batch_labels = []
            batch_data = []

        # if (episode % 100) == 0:
        #     print("episode: " + str(episode))

        episode += 1


total_correct = 0
total_examples = 0

with open('test_mini_labeled.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        try:
            label = int(row[0])
        except:
            continue
        try:
            data = list(map(int,row[1:]))
        except:
            continue
        data = np.array(data)

        data = agent.binarize(data)

        total_examples += 1
        total_correct += (agent.predict(data) == label)

csvFile.close()

print("Correct predictions: " + str(total_correct) + "/" + str(total_examples))

print("Accuracy: " + str(np.round(total_correct / total_examples * 100, 2))
      + "%")
