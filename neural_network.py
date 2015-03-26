"""Learning and prediction functions for artificial neural networks."""

import collections
import common
import math
import random
import sys

# Throughout this file, layer 0 of a neural network is the inputs, layer 1
# is the first hidden layer, etc.; the last layer is the outputs.

class NeuralNetwork:
  """An artificial neural network.

  Fields:
    weights: a list of lists of lists of numbers, where
       weights[a][b][c] is the weight into unit b of layer a+1 from unit c in
         layer a
    num_hidden_layers: an integer, the number of hidden layers in the network
  """

  def __init__(self, weights=None):
    self.weights = weights
    if weights:
      self.num_hidden_layers = len(weights) - 1

  def get_unit_values(self, features):
    """Calculate the activation of each unit in a neural network.

    Args:
      features: a vector of feature values

    Returns:
      units, a list of lists of numbers, where
        units[a][b] is the activation of unit b in layer a
    """
    # COMPLETE THIS IMPLEMENTATION
    units = []
    units.append([])
    for k in range(len(features)):
      units[0].append(features[k])
    for i in range(len(self.weights)):
      units.append([])
      for j in range(len(self.weights[i])):
        total = 0
        for k in range(len(units[i])):
          val = self.weights[i][j][k]
          val *= units[i][k]
          total += self.weights[i][j][k] * units[i][k]
        units[i+1].append(self.activation(total))
    return units


  def predict(self, features):
    """Calculate the prediction of a neural network on one example

    Args:
      features: a vector of feature values

    Returns:
      A list of numbers, the predictions for each output of the network
          for the given example.
    """
    # COMPLETE THIS IMPLEMENTATION
    unit = self.get_unit_values(features)
      
    return unit[len(unit)-1]


  def calculate_errors(self, unit_values, outputs):
    """Calculate the backpropagated errors for an input to a neural network.

    Args:
      unit_values: unit activations, a list of lists of numbers, where
        unit_values[a][b] is the activation of unit b in layer a
      outputs: a list of correct output values (numbers)

    Returns:
      A list of lists of numbers, the errors for each hidden or output unit.
          errors[a][b] is the error for unit b in layer a+1.
    """
    # COMPLETE THIS IMPLEMENTATION
    err = [[] for i in range(len(unit_values)-1)]
    for i in range(len(outputs)):
      p = unit_values[len(unit_values)-1][i]
      errout = p * (1-p) * (outputs[i] - p)
      err[len(err)-1].append(errout)

    for i in reversed(range(len(unit_values)-2)):
      for j in range(len(unit_values[i+1])):
        error = 0
        for k in range(len(err[i+1])):
          p = unit_values[i+1][j]
          error += p * (1-p) * self.weights[i+1][k][j] * err[i+1][k]
        err[i].append(error)
    
    return err

  def activation(self, v):
    return 1 / (1 + math.exp(-v))

  def learn(self,
      data,
      num_hidden=16,
      max_iterations=1000,
      learning_rate=1,
      num_hidden_layers=1):
    """Learn a neural network from data.

    Sets the weights for a neural network based on training data.

    Args:
      data: a list of pairs of input and output vectors, both lists of numbers.
      num_hidden: the number of hidden units to use.
      max_iterations: the max number of iterations to train before stopping.
      learning_rate: a scaling factor to apply to each weight update.
      num_hidden_layers: the number of hidden layers to use.
        Unless you are doing the extra credit, you can ignore this parameter.

    Returns:
      This object, once learned.
    """
    # COMPLETE THIS IMPLEMENTATION
    # Use predict, get_unit_values, and calculate_errors
    # in your implementation!

    self.weights = []
    self.weights.append([])
    for i in range(num_hidden):
      self.weights[0].append([])
      for j in range(len(data[0][0])):
        self.weights[0][i].append(random.random()*1.2)

    for i in range(num_hidden_layers):
      self.weights.append([])
      for j in range(num_hidden_layers):
        self.weights[i+1].append([])
        for k in range(num_hidden):
          self.weights[i+1][j].append(random.random()*1.2)

    count = 0
    con = False
    while not con:
      count += 1
      con = True
      if count == max_iterations:
        break
      for i in range(len(data)):
        unit = self.get_unit_values(data[i][0])
        error = self.calculate_errors(unit, data[i][1])
        p = self.predict(data[i][0])
        for j in range(len(p)):
          p[j] = 0 if p[j] <= 0.5 else 1
        if not p == data[i][1]:
          con = False
        else:
          continue
          
        for j in range(len(self.weights)):
          for k in range(len(error[j])):
            for q in range(len(unit[j])):
              self.weights[j][k][q] += learning_rate * error[j][k] * unit[j][q]

    return self
