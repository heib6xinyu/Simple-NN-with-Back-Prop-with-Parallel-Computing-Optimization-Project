# Simple NN with Back Propagation and Parallel Computing Optimization
## Introduction
Welcome to our Neural Network Project, a basic platform designed to empower users in constructing and training neural networks with backpropagation. It offers a comprehensive suite of tools and functionalities catering to a wide range of neural network applications.

Our program stands out for its flexibility in building neural networks. Users can easily construct networks with varying architectures and depths to suit their specific problem domains. The core of our project lies in its robust training capabilities. We support various training options, including stochastic, minibatch, and batch gradient descent, allowing users to choose the method that best fits their dataset size and computational constraints.

Diversity in loss functions is another hallmark of our platform. With options like L1_norm, L2_norm, SVM (Support Vector Machine), and Softmax, users can tailor the learning process to their specific tasks, whether it be classification, regression, or other complex predictive modeling challenges.

Understanding the importance of data in machine learning, our project includes a robust data preprocessing module for two common classification datasets: iris.data and agaricus-lepiota.data. This module assists in transforming raw input data into a format compatible with neural networks using one-hot encoding. Additionally, our system robustly handles and rejects illegal data, ensuring data integrity and reliability in the training process.

In the realm of algorithm optimization, we have implemented commonly used techniques such as RMSprop, Nesterov Momentum, and Adam. These algorithms optimize the training process, ensuring faster convergence and improved performance of the neural networks.

Lastly, we have focused on optimizing neural network performance, including the use of OpenMP pragmas for parallel computing techniques.





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
1. Clone the repository: `git clone <repo-url>`


## Usage

### Quick Start Guide

Getting started with our Neural Network Project is straightforward and requires only a few steps to see the project in action. Below is a quick guide to compiling and running basic tests and examples using the provided shell scripts.

#### Compiling the Project

We have provided several shell scripts to compile different parts of the project. Depending on what you want to test or demonstrate, you can use one of the following scripts to compile the corresponding part of the project:

1. **Compile Basic Tests**: To compile the basic tests, run the following command in your terminal:

   ```bash
   ./compile_basictests.sh
   ```

2. **Compile PA11 Tests**: To compile PA11 tests, use:

   ```bash
   ./compile_pa11tests.sh
   ```

3. **Compile PA12 Tests**: For compiling PA12 tests, execute:

   ```bash
   ./compile_pa12tests.sh
   ```

4. **Compile PA14 Gradient Descent**: If you want to compile the PA14 gradient descent part, run:

   ```bash
   ./compile_pa14gd.sh
   ```

5. **Compile All Tests**: To compile all tests at once, use:

   ```bash
   ./compile_tests.sh
   ```

Each script will compile the respective parts of the project and prepare them for execution.

#### Running the Project

After compilation, you can run the corresponding parts of the project to see them in action. Use the following scripts to run the different sections:

1. **Run Basic Tests**:

   ```bash
   ./run_basictests.sh
   ```

2. **Run PA11 Tests**:

   ```bash
   ./run_pa11tests.sh
   ```

3. **Run PA12 Tests**:

   ```bash
   ./run_pa12tests.sh
   ```

4. **Run PA14 Gradient Descent**:

   ```bash
   ./run_pa14gd.sh
   ```

Each script executes the compiled binaries and demonstrates the functionalities of the respective parts of the project. These scripts provide a hands-on way to observe the neural network’s behavior, training process, and the results of different configurations and optimizations.

### Detailed Usage

To run the neural network gradient descent with specific parameters, our program offers a flexible command-line interface. Below is an example command and a detailed explanation of each parameter:

#### Command Format

```bash
./PA14GD <data set> <gradient type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <adaptive technique> <decay rate> <epsilon> <beta1> <beta2> <layer sizes...>
```

#### Example Usage

```bash
./PA14GD mushroom minibatch 20 softmax 100 0.1 0.01 0.9 adam 0.96 0.0000001 0.9 0.999 10 10
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
  
- **Gradient Descent Scripts (`PA14GD.cpp`)**: These scripts integrate all the components mentioned above. They implement the gradient descent algorithm and orchestrate the training process of the neural network, using various parameters and configurations specified by the user.
  
### Function Descriptions


### Case Studies
Detailed examples or tutorials for specific tasks or datasets.



## Contact Information

### Authors
List of authors and maintainers with contact information.


## Appendices

### References
List of references and influential sources.

### Acknowledgments
Acknowledgments to contributors, sponsors, etc.
