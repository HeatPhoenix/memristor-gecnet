import numpy as np
import heyoka as hy

class heyoka_ffnn:
    """This helper class facilitates the interface between heyoka.py API and pytorch API
       It allows to train simple ffnn networks in torch (softplus) and use them in the r.h.s. or event equations in heyoka
    """
    def __init__(self, n_in, n_out, n_hidden, n_neurons_per_layer):  
        # Number of inputs
        self.n_in = n_in
        # Number of outputs
        self.n_out = n_out
        # Number of neuronal layers (counting input and output) 
        self.n_layers = n_hidden + 2
        # Number of neurons per neuronal layer
        self.n_neurons = [n_in] + [n_neurons_per_layer]*n_hidden + [n_out]
        # Number of weight parameters
        self.n_w = 0
        # Weights
        self.weight = [0.] * (self.n_layers - 1)
        # Number of bias parameters
        # Biases
        self.bias = [0.] * (self.n_layers - 1)
        self.n_b = 0
        for i in range(self.n_layers - 1):
            # weights
            self.n_w += self.n_neurons[i] * self.n_neurons[i + 1]
            # biases
            self.n_b += self.n_neurons[i + 1]
        # This contains the weights and biases flattened in one array (as requested by heyoka.py parameter machinery)
        self.par = np.zeros(self.n_w + self.n_b)
        
    def softplus(self, x):
        return np.log(1+np.exp(x))
  
    def __call__(self, inputs):
        """ Evaluates the NN output from one input 
            (used mainly to check the network is indeed equivalent to the one trained in pytorch)
        """
        # First hidden layer
        retval = self.softplus(np.dot(self.weight[0],inputs)+self.bias[0].reshape((-1,1)))
        
        for i in range(1,  self.n_layers - 2):
            retval = self.softplus(np.dot(self.weight[i],retval)+self.bias[i].reshape((-1,1)))
        # Linear output layer
        retval =  np.dot(self.weight[-1], retval)+ self.bias[-1].reshape((-1,1))
        
        return retval
        
    def compute_heyoka_expression(self, inputs):
        """ Computes the network outputs as heyoka expressions. For deep networks l>=4 it will fail.
            as the explicit expressions become too large.
        """
        from copy import deepcopy
        assert(len(inputs) == self.n_in)
        retval = deepcopy(inputs)
        for i in range(1,  self.n_layers):
            retval = self.compute_heyoka_layer(i, retval)
        return retval

    # from the layer, the neuron and the input,
    # returns the flattened index of the corresponding weight
    def flattenw(self, layer, neuron, inp):
        assert(layer > 0)
        counter = 0
        for k in range(1, layer):
            counter += self.n_neurons[k] * self.n_neurons[k - 1]
        counter += neuron * self.n_neurons[layer - 1]
        return counter + inp

    # from the layer, the neuron and the input,
    # returns the flattened index of the corresponding bias
    def flattenb(self, layer, neuron):
        assert(layer > 0)
        counter = 0
        for k in range(1, layer):
            counter += self.n_neurons[k]
        return counter + neuron + self.n_w

    def compute_heyoka_layer(self, layer, ins):
        assert(layer > 0)
        assert(len(ins) == self.n_neurons[layer - 1]) # Check len(inputs) == number of neurons in preciding layer
        retval = [0] * self.n_neurons[layer]  # Initiate output
        for neuron in range(self.n_neurons[layer]):  # Run through neurons
            # wij xj
            tmp = []
            for inputs in range(self.n_neurons[layer - 1]): # Run through inputs
                tmp.append(hy.par[self.flattenw(layer, neuron, inputs)] * ins[inputs])
            # b
            tmp.append(hy.par[self.flattenb(layer, neuron)])
            # wij xj + bi
            retval[neuron] = hy.sum(tmp)
            # non linearity
            if layer < self.n_layers-1: # Last layer has linear activation function
                retval[neuron] = hy.log(1+hy.exp(retval[neuron]))
        return retval

    def set_parameters_from_torch(self, state_dict):
        """From a state_dict of a pytorch network (converted to numpy) it sets
            weight, bias and par of the ffnn
        """
        for key in state_dict:
            idx = int(key.split(".")[0]) // 2 ## assuming nonlinearities layers always after a linear layer
            what = key.split(".")[1]
            if what == "weight":
                self.weight[idx] = state_dict[key]
            else:
                self.bias[idx] = state_dict[key]
        for l in range(1, self.n_layers):
            for n in range(self.n_neurons[l]):
                for i in range(self.n_neurons[l - 1]):
                    self.par[self.flattenw(l, n, i)] = self.weight[l-1][n,i]
        for l in range(1, self.n_layers):
            for n in range(self.n_neurons[l]): 
                self.par[self.flattenb(l, n)] = self.bias[l-1][n]