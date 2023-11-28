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
Simple example to get the project running.

### Detailed Usage
Explanation of how to use the main features with code snippets and examples.

### Command Line Interface
If applicable, document the CLI commands and arguments.

## Code Documentation

### Code Structure
Overview of the codebase structure and explanation of key directories and files.

### Modules and Classes
Description of major classes and modules, their functions, and interactions.

### Function Descriptions
Details of key functions including parameters, return values, and examples.

## Neural Network Details

### Architecture
Description of the neural network architecture(s), layers, activation functions, loss functions, etc.

### Training Process
Details of the training process, optimization algorithms, batch size, epochs, etc.

### Data Handling
Explanation of data processing, loading, and usage in the model.

## Examples and Tutorials

### Case Studies
Detailed examples or tutorials for specific tasks or datasets.

### Jupyter Notebooks
Link to Jupyter Notebooks demonstrating the project's practical usage.

## Contributing

### Guidelines
Guidelines for contributing to the project.

## License
Specify the license under which the project is released.

## Contact Information

### Authors
List of authors and maintainers with contact information.

### Support
Information on where and how to get support (e.g., issue tracker, discussion forum).

## Appendices

### References
List of references and influential sources.

### Acknowledgments
Acknowledgments to contributors, sponsors, etc.
