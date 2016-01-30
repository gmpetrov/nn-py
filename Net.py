from Neuron import Neuron
import math

'''
    class Net
'''
class Net:
    def __init__(self, topology):
        self.layers = []
        self.error = 0.0

        # Get the number of layers
        self.numLayers = len(topology)
        
        # Create layers
        for j in range(0, self.numLayers):
            self.layers.append([])
            
            numberOutputs = 0 if j == self.numLayers - 1 else topology[j + 1]
            
            # adding neuron to the last created layer
            for i in range(0, topology[j] + 1):
                self.layers[j].append(Neuron(numberOutputs + 1, i))

            # Set the bias neuron output value to constant 1.0
            self.layers[j][len(self.layers[j]) - 1].outputVal = 1.0
            #print str(self.layers[j])

    def __repr__(self):
        return "Layers : \n" + str(self.layers) + "\n"
    def __str__(self):
        return self.__repr__()

    def feedForward(self, inputVals):
        assert len(inputVals) == len(self.layers[0]) - 1, "Input values should have the same number than the numnber of neurons in the input layer"

        # assing the input values to neurons in the input layer
        for i in range(0, len(inputVals)):
            self.layers[0][i].outputVal = inputVals[i]

        # Forward propagate
        for i in range(1, self.numLayers):
            prevLayer = self.layers[i - 1]
            for n in range(0, len(self.layers[i]) - 1):
                self.layers[i][n].feedForward(prevLayer)
                            
    def backPropagation(self, targetVals):
        # Calculate the overall net error (RMS of output neuron errors)
        # The RMS (Root Mean Square) is what the network is trying to minimise
        self.error = 0.0
        outputLayer = self.layers[self.numLayers - 1]

        for i in range(0, len(outputLayer) - 1):
            delta = targetVals[i] - outputLayer[i].outputVal
            self.error += delta * delta
        self.error /= (len(outputLayer) - 1)
        self.error = math.sqrt(self.error) # RMS

        # Calculate output layer gradients
        for i in range(0, len(outputLayer) - 1):
            outputLayer[i].calcOutputGradients(targetVals[i])

        # Calculate gradients  on hiddent layers
        for j in range(self.numLayers - 2, 0, -1):
            hiddenLayer = self.layers[j]
            nextLayer = self.layers[j + 1]

            for i in range(0, len(hiddenLayer)):
                hiddenLayer[i].calcHiddenGradients(nextLayer)

        # For all layers form outputs to first hidden layer,
        # update connection weights
        for i in range(self.numLayers - 1, 0, -1):
            layer = self.layers[i]
            prevLayer = self.layers[i - 1]

            for j in range(0, len(layer) - 1):
                layer[j].updateInputWeights(prevLayer);


    def getResults(self):
        resultVals = []
        outputLayer = self.layers[self.numLayers - 1]
        for i in range(0, len(outputLayer) - 1):
            resultVals.append(outputLayer[i].outputVal)
        return resultVals


