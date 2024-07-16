"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from convnet_pytorch import ConvNet
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  # one-hot to class
  _, predicted_classes = torch.max(predictions, 1)
  _, target_classes = torch.max(targets, 1)
  correct_predictions = (predicted_classes == target_classes).sum().item()
  accuracy = correct_predictions / targets.size(0)

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  torch.manual_seed(42) # ADD torch SEED

  # Load data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  train_loader = cifar10['train']
  test_loader = cifar10['test']

  # Initialize model, loss function, and optimizer
  model = ConvNet(3, 10)  # Table 1. 3 of the first layer input, 10 of the last layer input 
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate) # Use Adam optimizer with default learning rate

  # Training loop
  for step in range(FLAGS.max_steps):
    model.train()

    # Get a batch of training data
    x, y = train_loader.next_batch(FLAGS.batch_size)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y).argmax(dim=1)
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training information
    if step % 100 == 0:
      print(f"Step [{step}/{FLAGS.max_steps}], Loss: {loss.item()}")

    # Evaluation on the test set
    if step % FLAGS.eval_freq == 0 or step == FLAGS.max_steps - 1:
      model.eval()
      total_accuracy = 0
      total_samples = 0
      num_batches = test_loader.num_examples // FLAGS.batch_size 
      
      with torch.no_grad():
        for _ in range(num_batches):
          x_test, y_test = test_loader.next_batch(FLAGS.batch_size)
          x_test = torch.tensor(x_test, dtype=torch.float32)
          y_test = torch.tensor(y_test)
          outputs_test = model(x_test)
          total_accuracy += accuracy(outputs_test, y_test) * x_test.size(0)
          total_samples += x_test.size(0)
      test_accuracy = total_accuracy / total_samples
      print(f"Step [{step}/{FLAGS.max_steps}], Test Accuracy: {test_accuracy}")

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()