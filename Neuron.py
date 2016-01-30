import random
from Tools import Tools

'''
    class Neuron
'''
class Neuron:
    def __init__(self, numberOutputs, index):
        self.outputVal = 0.0
        self.index = index
        self.gradient = 0.0
        self.learningRate = 0.15
        self.alpha = 0.1
        self.tools = Tools()

        # for the neuron thas this neuron will feeds
        # Array of Connections
        self.outputWeights = []

        for i in range(0, numberOutputs):
            self.outputWeights.append(Connection())
            self.outputWeights[i].weight = self.randomWeight()
            #print self.outputWeights[i].weight

    def __repr__(self):
        return "Neuron : index = " + str(self.index) + ", outputVal = " + str(self.outputVal) + "\n" # + ", Connections : " + str(self.outputWeights) 
    def __str__(self):
        return self.__repr__()
    
    def feedForward(self, prevLayer):
        # Sum the previous layer's ouputs (which are our inputs)
        # Include the bias node from the previous layer

        sum = 0.0
    
        for i in range(0, len(prevLayer)):
            sum += prevLayer[i].outputVal * prevLayer[i].outputWeights[self.index].weight

        # update the outputVal
        self.outputVal = self.activationFunction(sum)
    
    def activationFunction(self, x):
        #return self.tools.sigmoid(x)
        return self.tools.tanh(x)

    def activationFunctionDerivative(self, x):
        #return self.tools.sigmoidDerivative(x)
        return self.tools.tanhDerivative(x)

    def randomWeight(self):
        return random.uniform(0.0, 1.0)

    def calcOutputGradients(self, targetVal):
        delta = targetVal - self.outputVal
        self.gradient = delta * self.activationFunctionDerivative(self.outputVal) 
    def calcHiddenGradients(self, nextLayer):
        dow = self.sumDow(nextLayer)
        self.gradient = dow * self.activationFunctionDerivative(self.outputVal)
    def sumDow(self, nextLayer):
        sum = 0.0

        for i in range(0, len(nextLayer) - 1):
            sum += (self.outputWeights[i].weight * nextLayer[i].gradient)
        return sum

    def updateInputWeights(self, prevLayer):
        for i in range(0, len(prevLayer)):
            neuron = prevLayer[i]
            oldDeltaWeight = neuron.outputWeights[self.index].deltaWeight
            newDeltaWeight = self.learningRate * neuron.outputVal * self.gradient + self.alpha * oldDeltaWeight
            neuron.outputWeights[self.index].deltaWeight = newDeltaWeight
            neuron.outputWeights[self.index].weight += newDeltaWeight
'''
    Class Connection
'''
class Connection:
    def __init__(self):
        self.weight = 0.0
        self.deltaWeight = 0.0
    def __repr__(self):
        return "Connection : weight = " + str(self.weight) + ", deltaWeight = " + str(self.deltaWeight) + "\n"
    def __str__(self):
        return __repr__()


