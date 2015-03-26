"""Learning and prediction functions for perceptrons."""

import common


class NotConverged(Exception):
  """An exception raised when the perceptron training isn't converging."""


class Perceptron:
  def __init__(self, weights=None):
    self.weights = weights

  def learn(self, examples, max_iterations=100):
    """Learn a perceptron from [([feature], class)].

    Set the weights member variable to a list of numbers corresponding
    to the weights learned by the perceptron algorithm from the training
    examples.

    The number of weights should be one more than the number of features
    in each example.

    Args:
      examples: a list of pairs of a list of features and a class variable.
        Features should be numbers, class should be 0 or 1.
      max_iterations: number of iterations to train.  Gives up afterwards

    Raises:
      NotConverged, if training did not converge within the provided number
        of iterations.

    Returns:
      This object
    """
    # COMPLETE THIS IMPLEMENTATION
    self.weights = [0 for i in range(len(examples[0][0])+1)]
    for count in range(max_iterations):
      good = True
      for (features, out) in examples:
        total = 0
        for i in range(len(features)):
          total += features[i] * self.weights[i]
        total += self.weights[len(self.weights)-1]
        if total > 0 and out == 0:
          for i in range(len(features)):
            self.weights[i] -= features[i]
          self.weights[len(self.weights)-1] -= 1
          good = False
        elif total <= 0 and out == 1:
          for i in range(len(features)):
            self.weights[i] += features[i]
          self.weights[len(self.weights)-1] += 1
          good = False
      if good:
        return self

    raise NotConverged
    return self

  def predict(self, features):
    """Return the prediction given perceptron weights on an example.

    Args:
      features: A vector of features, [f1, f2, ... fn], all numbers

    Returns:
      1 if w1 * f1 + w2 * f2 + ... * wn * fn + t > 0
      0 otherwise
    """
    # COMPLETE THIS IMPLEMENTATION
    total = 0
    for i in range(len(features)):
      total += features[i] * self.weights[i]
    total += self.weights[len(self.weights)-1]
    if total > 0:
      return 1
    return 0
