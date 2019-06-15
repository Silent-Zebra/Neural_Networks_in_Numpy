import csv
import numpy as np


class OneLayerNN:
    def __init__(self, neurons: int = 10, input_size: int = 784,
                 learn_rate: float = 0.01):
        self.neurons = neurons
        self.input_size = input_size
        self.weights = np.random.rand(neurons, input_size) / np.math.sqrt(input_size)
        self.learn_rate = learn_rate

    def binarize(self, input_vector: np.array):
        new_vector = input_vector > 0
        return new_vector

    def _sigmoid(self, vector: np.array):
        new_vector = 1 / (1 + np.exp(- vector))
        return new_vector

    def _prediction_vector(self, input_vector: np.array):
        # Should be a 10x1 list output
        values = (self.weights @ input_vector)
        values = self._sigmoid(values)
        return values

    def predict(self, input_vector: np.array):
        vector = self._prediction_vector(input_vector)
        return np.argmax(vector)

    def _generate_label_vector(self, label):
        out = np.zeros((self.neurons))
        out[label] = 1
        return out

    def gd(self, batch_labels, batch_data):
        accumulated_errors = np.zeros((self.neurons, self.input_size))

        for i in range(len(batch_labels)):
            input_vector = batch_data[i]
            vector = self._prediction_vector(input_vector)
            label = batch_labels[i]
            target = self._generate_label_vector(label)

            delta = vector * (1 - vector) * (target - vector)

            accumulated_errors += np.outer(delta, input_vector)
            # Above helps speed up computation compared to for loop such as:
            # for i in range(self.neurons):
            #     accumulated_errors[i] += delta[i] * input_vector

        self.weights += self.learn_rate * accumulated_errors


agent = OneLayerNN(10)

batch_size = 100

epochs = 20

full_data = []

with open('train_1000.csv', 'r') as csvFile:

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
