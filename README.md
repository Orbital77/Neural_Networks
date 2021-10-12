# Neural_Networks

# Objective
  This project aims to create a functional and easy to implement deep learning model using the C programming language.
Modules may eventaully be created in python to help format input data, but the main program will use C.

# Progress
  Currently, the development build in 'Neural_Networks/C/machine_learning/dev' is working. Our program is now semi-complete, with several new features:
   full support for one-hot encoding,
   full MLP (multi-layer-perceptron) with foward and backward propagation complete with stochastic gradient descent,
   file I/O for loading/saving networks,
   the ability to easily change activation function per layer through a config file,
   and a designated training function that trains the network for a given amount of epochs
  
# Bugs
  The program only has one major problem at the moment. The issue is random (probably caused by random initialization of weights), and only appears at epoch ~4000-9000 every 10 runs of the program or so. This problem will be fixed before any other additional features are added.
  
