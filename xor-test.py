from nn import Net
import random


def xorTest():

    # Conf of the net
    config = [2, 3, 1]

    nn = Net(config)

    for i in range(0, 5000):
        # Generate dataset
        a = 0.0 if random.uniform(0.0, 1.0) <= 0.5 else 1.0
        b = 0.0 if random.uniform(0.0, 1.0) > 0.5 else 1.0

        # Target value
        res = int(a) ^ int(b)

        inputVals = []
        inputVals.append(a)
        inputVals.append(b)

        targetVals = []
        targetVals.append(res)

        nn.feedForward(inputVals)
        print "Inputs : " + str(inputVals)

        nn.backPropagation(targetVals)
        print "Target : " + str(targetVals[0])

        resultVals = []
        resultVals = nn.getResults()
        print resultVals
        print "========="
xorTest()
