import numpy as np

#Codes from Adrian's Book

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha


        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))