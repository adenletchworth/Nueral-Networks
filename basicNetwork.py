# we have these inputs for our model
inputs = [1,3,4]

# we have these three nuerons each with 3 weights
neurons = [[.5,.1,.7],[.3,.4,-.2],[0,.1,.3]]
# we have the associated bias for each neuron
biases = [3,1,2]
# initalie outputs arrays
outputs = []
# loop through all of our weights
for weights, bias in zip(neurons,biases):
    # refresh neuron
    product = 0
    # get linear combination of weights and input
    for weight, inp in zip(weights,inputs):
        product += weight * inp
    # add it to the output list
    outputs.append(product+bias)
# check our output list
print(outputs)