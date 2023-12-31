# Simple NN with Back Propagation and Parallel Computing Optimization
## Introduction
Welcome to our Neural Network Project, a basic platform designed to empower users in constructing and training neural networks with backpropagation. It offers a comprehensive suite of tools and functionalities catering to a wide range of neural network applications.

Our program stands out for its flexibility in building neural networks. Users can easily construct networks with varying architectures and depths to suit their specific problem domains. The core of our project lies in its robust training capabilities. We support various training options, including stochastic, minibatch, and batch gradient descent, allowing users to choose the method that best fits their dataset size and computational constraints.

Diversity in loss functions is another hallmark of our platform. With options like SVM (Support Vector Machine), and Softmax, users can tailor the learning process to their specific tasks, whether it be classification, regression, or other complex predictive modeling challenges.

Understanding the importance of data in machine learning, our project includes a robust data preprocessing module for two common classification datasets: iris.data and agaricus-lepiota.data. This module assists in transforming raw input data into a format compatible with neural networks using one-hot encoding. Additionally, our system robustly handles and rejects illegal data, ensuring data integrity and reliability in the training process.

In the realm of algorithm optimization, we have implemented commonly used techniques such as RMSprop, Nesterov Momentum, and Adam. These algorithms optimize the training process, ensuring faster convergence and improved performance of the neural networks.

## Rubric Points
- Cross-platform compilation - +2pts.
- Illegal Input Handling - +2pts.
- Non-trivial optimization/techniques - +6 pts.
   (Stochastic, Minibatch, batch, adam, rmsprop, Nesterov Momentum)
- Documentation - +2pts.
- Benchmarking with baselines - +5pts.


## Background
The data we used in this project is iris.data and agaricus-lepiota.data.

## Installation and Setup

### Dependencies
- g++ compiler

#### On Linux:

1. **Install GCC**:
   - Open a terminal.
   - Update your package list (on Debian/Ubuntu systems):
     ```bash
     sudo apt-get update
     ```
   - Install GCC:
     ```bash
     sudo apt-get install g++
     ```
   - This will install both the C (`gcc`) and C++ (`g++`) compilers.

2. **Verify Installation**:
   - After installation, you can verify it by running:
     ```bash
     g++ --version
     ```
   - This should show the version of GCC that was installed.

#### On macOS:

1. **Install Xcode Command Line Tools**:
   - Open a terminal.
   - Install Xcode Command Line Tools, which includes GCC:
     ```bash
     xcode-select --install
     ```
   - Follow the prompts to complete the installation.

2. **Verify Installation**:
   - You can verify the installation with:
     ```bash
     g++ --version
     ```

#### On Windows:

1. **Install MinGW** (Minimalist GNU for Windows):
   - Download MinGW from [here](http://www.mingw.org/).
   - During installation, make sure to select the `mingw32-gcc-g++` package.
   - Follow the installation prompts.

2. **Add to PATH**:
   - After installation, add the MinGW bin directory to your system's PATH. This is typically found in `C:\MinGW\bin`.
   - You can add it to the PATH by searching for "Environment Variables" in Windows and editing the `Path` variable under "System variables" to include the MinGW bin path.

3. **Verify Installation**:
   - Open Command Prompt and type:
     ```cmd
     g++ --version
     ```

After completing these steps, you should be able to compile C++ programs using `g++` in your terminal or command prompt.
### Installation Guide
1. Clone the repository: `git clone https://github.com/heib6xinyu/Simple-NN-with-Back-Prop-with-Parallel-Computing-Optimization-Project.git`


## Usage

### Quick Start Guide

Getting started with our Neural Network Project is straightforward and requires only a few steps to see the project in action. Below is a quick guide to compiling and running basic tests and examples using the provided shell scripts.

#### Compiling the Project

We have provided several shell scripts to compile different parts of the project. Depending on what you want to test or demonstrate, you can use one of the following scripts to compile the corresponding part of the project:

1. **Compile Basic Tests**: To compile the basic tests, run the following command in your terminal:

   ```bash
   ./compile_basictests.sh
   ```

2. **Compile NN Tests**: To compile NN tests, use:

   ```bash
   ./compile_nntests.sh
   ```

3. **Compile Gradient Tests**: For compiling Gradient tests, execute:

   ```bash
   ./compile_gradienttests.sh
   ```

4. **Compile Gradient Descent**: If you want to compile the gradient descent part, run:

   ```bash
   ./compile_gd.sh
   ```

5. **Compile All Tests**: To compile all tests at once, use:

   ```bash
   ./compile_all.sh
   ```

Each script will compile the respective parts of the project and prepare them for execution.

#### Running the Project

After compilation, you can run the corresponding parts of the project to see them in action. Use the following scripts to run the different sections:

1. **Run Basic Tests**:

   ```bash
   ./run_basictests.sh
   ```

2. **Run NN Tests**:

   ```bash
   ./run_nntests.sh
   ```

3. **Run Gradient Tests**:

   ```bash
   ./run_gradienttests.sh
   ```

4. **Run Gradient Descent**:

   ```bash
   ./run_gd.sh
   ```

Each script executes the compiled binaries and demonstrates the functionalities of the respective parts of the project. These scripts provide a hands-on way to observe the neural network’s behavior, training process, and the results of different configurations and optimizations.

### Detailed Usage

To run the neural network gradient descent with specific parameters, our program offers a flexible command-line interface. Below is an example command and a detailed explanation of each parameter:

#### Command Format

```bash
./GradientDescent <data set> <gradient type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <adaptive technique> <decay rate> <epsilon> <beta1> <beta2> <layer sizes...>
```

#### Example Usage

```bash
./GradientDescent mushroom minibatch 20 softmax 100 0.1 0.01 0.9 adam 0.96 0.0000001 0.9 0.999 10 10
```

This command runs gradient descent on the mushroom dataset with the following configuration:

- **Data Set**: `mushroom` - Specifies the mushroom dataset as input.
- **Gradient Type**: `minibatch` - Uses minibatch gradient descent.
- **Batch Size**: `20` - Sets the batch size to 20.
- **Loss Function**: `softmax` - Employs the softmax loss function for the training process.
- **Epochs**: `100` - The model will train for 100 epochs.
- **Bias**: `0.1` - Initializes node biases to 0.1.
- **Learning Rate**: `0.01` - Sets the learning rate to 0.01.
- **Mu**: `0.9` - Specifies the mu (momentum) parameter for the gradient descent.
- **Adaptive Technique**: `adam` - Uses the Adam optimization algorithm.
- **Decay Rate**: `0.96` - Sets the decay rate for the optimizer to 0.96.
- **Epsilon (ϵ)**: `0.0000001` - The epsilon parameter for preventing division by zero in the Adam optimizer.
- **Beta1**: `0.9` - Sets the Beta1 parameter for the Adam optimizer.
- **Beta2**: `0.999` - Sets the Beta2 parameter for the Adam optimizer.
- **Layer Sizes**: `10 10` - Configures the network with two hidden layers, each containing 10 nodes.

#### Explanation

This command demonstrates how to run the neural network with a specific set of hyperparameters and configurations. You can adjust these parameters according to your requirements to experiment with different network behaviors and training dynamics. The flexibility in parameter specification allows for extensive experimentation and fine-tuning, catering to various data characteristics and training needs.


## Code Documentation

### Code Structure

Our Neural Network project is organized into several key components, each residing in its own directory and serving a specific function in the system. Here's an overview of the project's code structure:

#### 1. Data Folder

This folder contains the core data handling components of our system.

#### 2. Datasets Folder

- **Datasets Directory**: This directory contains the actual datasets used by the neural network, such as `iris.data` and `agaricus-lepiota.data`. These datasets are utilized by the `DataSet` class for training and testing the neural network.

#### 3. Network Folder

This folder encompasses the components that constitute the neural network.

#### 4. Util Folder

This folder contains utility functions and classes that support various operations of the neural network.


#### 5. Gradient Descent Scripts

Each part of the codebase plays a pivotal role in the functioning of the neural network, from data handling to network construction, training, and utility support. This modular approach not only organizes the code effectively but also enhances maintainability and scalability.

### Modules and Classes
- **`Instance.cpp` and `Instance.h`**: These files define the `Instance` class, responsible for representing individual data instances or samples. This class includes functionalities for handling input data, including one-hot encoding and managing data attributes.

- **`DataSet.cpp` and `DataSet.h`**: These files define the `DataSet` class, which works in conjunction with the `Instance` class. The `DataSet` class is responsible for processing and storing a collection of instances, handling tasks such as calculating means, standard deviations, and other dataset-level operations.
- **`Edge.cpp` and `Edge.h`**: Define the connections or 'edges' between nodes in the neural network.

- **`Node.cpp` and `Node.h`**: Represent the nodes or 'neurons' of the network.

- **`NeuralNetwork.cpp` and `NeuralNetwork.h`**: The core file that integrates nodes and edges to form the complete neural network.

- **Supporting Definitions**: Includes definitions of `ActivationType`, `LossFunction`, and `NodeType`, which are essential for specifying the behavior and characteristics of the neural network.

- **`Log.cpp` and `Log.h`**: Implement logging functionalities, crucial for monitoring and debugging the system.

- **`Vector.cpp` and `Vector.h`**: Provide vector-related utility functions, useful in various mathematical and data processing operations.

- **Test Scripts**: Includes functions for conducting various tests on the neural network to ensure its correctness and efficiency.
  
- **Gradient Descent Scripts (`GradientDescent.cpp`)**: These scripts integrate all the components mentioned above. They implement the gradient descent algorithm and orchestrate the training process of the neural network, using various parameters and configurations specified by the user.
  
### Function Descriptions
#### DataSet Class

##### Constructor
- `DataSet::DataSet(const std::string& name, const std::string& filename)`: Constructs a new `DataSet` object using a specified file. It initializes the `DataSet` with a `name` and loads data from `filename`, parsing each line to create `Instance` objects.

##### Data Normalization and Processing
- `std::vector<double> DataSet::getInputMeans()`: Calculates and returns the mean of each input column in the data set.
- `std::vector<double> DataSet::getInputStandardDeviations()`: Computes the standard deviations for each input column.
- `void DataSet::normalize(const std::vector<double>& inputMeans, const std::vector<double>& inputStandardDeviations)`: Normalizes the data set by subtracting the mean and dividing by the standard deviation for each input.

##### Data Retrieval and Management
- `std::string DataSet::getName() const`: Returns the name of the data set.
- `size_t DataSet::getNumberInstances() const`: Provides the total number of instances in the data set.
- `int DataSet::getNumberInputs() const`: Returns the number of inputs per instance.
- `int DataSet::getNumberOutputs() const`: Indicates the number of outputs per instance.
- `int DataSet::getNumberClasses() const`: Retrieves the number of unique classes in the data set.
- `void DataSet::shuffle()`: Randomizes the order of instances in the data set.
- `Instance DataSet::getInstance(int position) const`: Retrieves a specific instance based on its position.
- `std::vector<Instance> DataSet::getInstances(int position, int numberOfInstances) const`: Obtains a subset of instances from a specified position for a given number of instances.
- `const std::vector<Instance>& DataSet::getInstances() const`: Returns all instances in the data set.

#### Instance Class 

##### Constructor
- `Instance::Instance(const std::vector<double>& expectedOutputs, const std::vector<double>& inputs)`: Creates an `Instance` object with given expected outputs and inputs. The constructor initializes the `Instance` with vectors of expected outputs and inputs.

##### Comparison Functions
- `bool Instance::equals(const std::vector<double>& otherExpectedOutputs, const std::vector<double>& otherInputs) const`: Compares this `Instance` to another set of expected outputs and inputs to determine if they are the same. It returns `true` if the provided expected outputs and inputs are the same as those in the `Instance`.
- `bool Instance::equals(const Instance& other) const`: Compares this `Instance` to another `Instance` object. It returns `true` if the expected outputs and inputs in both instances are the same.

##### String Representation
- `std::string Instance::toString() const`: Generates a nicely readable string from this `Instance`. It constructs a string representation showing the expected outputs and inputs of the `Instance`.

#### Edge Class


##### Constructor
- `Edge::Edge(Node* inputNode, Node* outputNode)`: Constructs a new edge in the neural network between the specified input and output nodes. It initializes the weight and delta to 0 and registers the edge with the input and output nodes.

##### Backward Propagation
- `void Edge::propagateBackward(double delta)`: Takes an incoming delta (error) from the output node and propagates it backward to the input node. It updates the `weightDelta` by multiplying the delta with the post-activation value of the input node and accumulates the deltas in the input node.

##### Weight Management
- `void Edge::setWeight(double new_weight)`: Sets a new weight for the edge.
- `double Edge::getWeight()`: Returns the current weight of the edge.

##### Equality Check
- `bool Edge::equals(Edge other) const`: Checks if two edges are equal by comparing their input and output nodes. Returns `true` if both the input and output nodes of the edges are the same.

##### String Representation
- `std::string Edge::toString()`: Generates a readable string representation of the edge, detailing its input and output nodes.

#### Node Class 

The `Node` class in C++ represents a node in a neural network. It maintains lists of input and output edges, its value, layer information, and type (input, hidden, or output). Below are the function descriptions adapted from the Java version to the C++ implementation:

##### Constructor
- `Node::Node(int layerValue, int numberValue, NodeType type, ActivationType actType)`: Creates a new node at a given layer in the network, specifying its type (input, hidden, or output) and the activation function to use.

##### Resetting Node State
- `void Node::reset()`: Resets the node's values and deltas needed for each forward and backward pass. Also resets the deltas for outgoing edges.

##### Edge Management
- `void Node::addOutgoingEdge(std::shared_ptr<Edge> outgoingEdge)`: Adds an outgoing edge to this node.
- `void Node::addIncomingEdge(std::shared_ptr<Edge> incomingEdge)`: Adds an incoming edge to this node.

##### Forward Propagation
- `void Node::propagateForward()`: Propagates the node's post-activation value to all its output nodes by applying the appropriate activation function.

##### Activation Functions
- `void Node::applyLinear()`: Applies the linear activation function to this node.
- `void Node::applySigmoid()`: Applies the sigmoid activation function to this node.
- `void Node::applyTanh()`: Applies the tanh activation function to this node.

##### Backward Propagation
- `void Node::propagateBackward()`: Propagates the delta/error back from this node to its incoming edges.

##### Weight Initialization
- `void Node::initializeWeightsAndBias(double newBias)`: Sets the node's bias and randomly initializes each incoming edge weight.

##### Weight and Delta Management
- `int Node::getWeights(int position, std::vector<double>& weights) const`: Gets the weights of this node and its outgoing edges.
- `int Node::getDeltas(int position, std::vector<double>& deltas)`: Gets the deltas of this node and its outgoing edges.
- `int Node::setWeights(int position, std::vector<double>& weights)`: Sets the weights of this node and its outgoing edges.

##### Additional Functions
- `std::string Node::toString() const`: Returns a concise string representation of the node.
- `std::string Node::toDetailedString() const`: Returns a detailed string representation of the node.

#### NeuralNetwork Class 

##### Constructor
- `NeuralNetwork::NeuralNetwork(int inputLayerSize, const std::vector<int>& hiddenLayerSizes, int outputLayerSize, LossFunction lossFunc)`: Constructs a neural network with specified sizes for the input layer, hidden layers, and output layer, along with a chosen loss function.

##### Weight Management
- `int NeuralNetwork::getNumberWeights() const`: Returns the total number of weights (including biases) in the neural network.
- `void NeuralNetwork::setWeights(std::vector<double>& newWeights)`: Sets the weights of the network to the values provided in `newWeights`.
- `std::vector<double> NeuralNetwork::getWeights() const`: Returns a vector containing all the weights of the network.
- `std::vector<double> NeuralNetwork::getDeltas() const`: Obtains the deltas (gradients) for all the weights in the network.

##### Network Configuration
- `void NeuralNetwork::connectFully()`: Fully connects all nodes in each layer to all nodes in the subsequent layer.
- `void NeuralNetwork::connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber)`: Connects a specific node in one layer to a specific node in another layer.

##### Initialization
- `void NeuralNetwork::initializeRandomly(double bias)`: Initializes the weights of the network randomly using a normal distribution and sets the biases of the nodes.

##### Forward and Backward Propagation
- `double NeuralNetwork::forwardPass(const Instance& instance)`: Performs a forward pass through the network using the provided instance.
- `double NeuralNetwork::forwardPass(const std::vector<Instance>& instances)`: Processes multiple instances through the network and returns the sum of their outputs.
- `void NeuralNetwork::backwardPass()`: Conducts a backward pass through the network, updating the deltas based on the error.

##### Accuracy and Output
- `double NeuralNetwork::calculateAccuracy(const std::vector<Instance>& instances)`: Calculates the accuracy of the network on a set of instances.
- `std::vector<double> NeuralNetwork::getOutputValues() const`: Retrieves the output values from the output layer of the network.

##### Gradient Computation
- `std::vector<double> NeuralNetwork::getNumericGradient(const Instance& instance)`: Calculates the numerical gradient for a single instance.
- `std::vector<double> NeuralNetwork::getNumericGradient(const std::vector<Instance>& instances)`: Computes the numerical gradient for a set of instances.
- `std::vector<double> NeuralNetwork::getGradient(const Instance& instance)`: Gets the gradient of the network for a given instance using backpropagation.
- `std::vector<double> NeuralNetwork::getGradient(const std::vector<Instance>& instances)`: Obtains the gradient for a list of instances, summing up individual gradients.



### BenchMarking
I will build a neural network that have the same structure as mine, using tensorflow, and benchmark the two.
For TensorFlow Model, check benchmark_mushroom.py and benchmark_iris.py.
For run result, check tensorflownn_mushroom.txt and ourmodel_mushroom.txt, tensorflownn_iris.txt and ourmodel_iris.txt.

#### TensorFlow Model Results for iris dataset:

- The TensorFlow model shows impressive performance, achieving an accuracy of 100% at the 7th epoch with slight fluctuation.
- The loss drops very close to 0 with slight fluctuation, indicating that the model has effectively learned to classify the data with little errors.
#### Our Model Results:
- Our model also achieves a high accuracy, it passes 95% in the 16th epoch, and stablized at approximately 99.33% in the 36th epoch.
- However, the loss (0.558782) does not decrease to zero. This could be due to several factors, such as differences in the architecture, optimization algorithm, or the way the loss is calculated and reported.
It's worth noting that while the accuracy is high, the loss not reducing to a level closer to zero as in the TensorFlow model might indicate some room for improvement.

#### Benchmarking Insights:

- Accuracy: Both models achieve high accuracy.
- Loss: The TensorFlow model effectively minimizes the loss to almost 0, which is not the case with our model. This discrepancy in loss values, despite high accuracy, might indicate differences in how the models handle classification boundaries or manage the error margins.

#### TensorFlow Model Results for mushroom dataset:

- The TensorFlow model shows impressive performance, achieving an accuracy of 100% very quickly and maintaining it throughout the training.
- The loss drops to zero, indicating that the model has effectively learned to classify the data with no errors.
#### Our Model Results:
- Our model also achieves a high accuracy of approximately 100% in the third epoch.
- However, the loss (0.313262) does not decrease to zero. This could be due to several factors, such as differences in the architecture, optimization algorithm, or the way the loss is calculated and reported.
It's worth noting that while the accuracy is high, the loss not reducing to a level closer to zero as in the TensorFlow model might indicate some room for improvement.

#### Benchmarking Insights:

- Accuracy: Both models achieve high accuracy.
- Loss: The TensorFlow model effectively minimizes the loss to zero, which is not the case with our model. This discrepancy in loss values, despite high accuracy, might indicate differences in how the models handle classification boundaries or manage the error margins.

#### Runtime Comparison:
The runtime comparison between our custom model and the TensorFlow model for two different datasets (Iris and Mushroom) shows some interesting differences in performance:

1. **Iris Dataset**:
   - **Our Model**: 0.972 seconds
   - **TensorFlow Model**: 10.846 seconds

   For the Iris dataset, our model is significantly faster than the TensorFlow model. This could be due to a number of factors such as the simplicity of our model, more efficient handling of smaller datasets, or less computational overhead compared to TensorFlow's more generalized and feature-rich framework.

2. **Mushroom Dataset**:
   - **Our Model**: 3 minutes 38.433 seconds
   - **TensorFlow Model**: 22.143 seconds

   For the Mushroom dataset, the situation is reversed, with our model taking substantially longer to run compared to TensorFlow. This difference could be due to several reasons:
   - **Dataset Complexity**: The Mushroom dataset is more complex and larger in size, which could mean our model is less efficient at handling such datasets compared to TensorFlow.
   - **Optimization and Scalability**: TensorFlow is highly optimized for handling larger and more complex datasets. Where our model lacks certain optimizations that become critical for larger datasets.
   - **Parallelization and Hardware Utilization**: TensorFlow is optimized to make the best use of available hardware (like GPUs and multi-core CPUs) for parallel processing, which could be a contributing factor to its faster performance on more demanding tasks.

  
## Appendices

### Acknowledgments and References
This C++ implementation of the Neural Network System is an extension of the original Java version developed as a part of DSCI640 Neural Networks with Professor Travis Desell (email: tjdvse@rit.edu) at Rochester Institute of Technology. I was introduced to the foundational structure of this system as a student in this class. The system was initially built for educational purposes, focusing on the basics of neural networks.

In its current form, the C++ builds off of the previous system to full develop a Deep Learning Framework. This addition is part of the coursework in CSCI-739: Topics in Intelligent Systems—Machine Learning Systems Implementation, under the instruction of Professor Weijie Zhao (email: wjz@rit.edu) at Rochester Institute of Technology. The aim is to enhance the system's capabilities and explore the practical application of machine learning algorithms in a more complex computing environment.

#### Contributors:
- Xinyu Hu (email: xh1165@rit.edu)
- Kevin Penkowski (email: kwp5892@rit.edu)

