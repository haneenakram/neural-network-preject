# Task 2: Multi-Layer Perceptron with Backpropagation

## 🧠 Project Overview

This project implements a **Multi-Layer Perceptron (MLP)** neural network using the **Backpropagation learning algorithm** for multi-class classification of bird species. The implementation supports flexible network architecture with customizable hidden layers and activation functions.

## 🎯 Algorithm: Backpropagation

### What is Backpropagation?
Backpropagation is a supervised learning algorithm that trains multi-layer neural networks by:
1. **Forward Pass**: Computing predictions through the network
2. **Error Calculation**: Measuring difference between prediction and target
3. **Backward Pass**: Propagating errors back through layers
4. **Weight Updates**: Adjusting weights to minimize error

### Mathematical Foundation
```
Forward Pass:
- z^(l) = W^(l) * a^(l-1) + b^(l)
- a^(l) = σ(z^(l))

Backward Pass:
- δ^(L) = ∇_a C ⊙ σ'(z^(L))
- δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))

Weight Updates:
- W^(l) = W^(l) - η * δ^(l+1) * (a^(l))^T
- b^(l) = b^(l) - η * δ^(l+1)
```

## 📊 Dataset: Birds Multi-Class Classification

### Dataset Specifications
- **Total Samples**: 150 birds (50 per species)
- **Species Classes**: A, B, C (3-class classification)
- **Features**: 5 dimensional input vector
  - `gender` (preprocessed to numerical)
  - `body_mass`
  - `beak_length` 
  - `beak_depth`
  - `fin_length`

### Data Split Strategy
- **Training**: First 30 samples from each class (90 total)
- **Testing**: Remaining 20 samples from each class (60 total)
- **Input Dimension**: 5 features
- **Output Dimension**: 3 classes (one-hot encoded)

## 🏗️ Network Architecture

### Flexible MLP Design
```
Input Layer (5 neurons)
    ↓
Hidden Layer 1 (n1 neurons) - User defined
    ↓
Hidden Layer 2 (n2 neurons) - Optional
    ↓
...
    ↓
Hidden Layer k (nk neurons) - User defined
    ↓
Output Layer (3 neurons) - Softmax
```

### Activation Functions
1. **Sigmoid**: σ(x) = 1/(1 + e^(-x))
   - Range: (0, 1)
   - Smooth gradient
   - Can suffer from vanishing gradient

2. **Hyperbolic Tangent**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
   - Range: (-1, 1)
   - Zero-centered
   - Stronger gradients than sigmoid

## 🖥️ GUI Interface

### User Configuration Options

#### Network Architecture
- **Number of Hidden Layers**: 1, 2, 3, ... (user defined)
- **Neurons per Layer**: Customizable for each hidden layer
- **Bias Option**: Include/exclude bias terms (checkbox)

#### Training Parameters
- **Learning Rate (η)**: Controls weight update step size
- **Number of Epochs (m)**: Maximum training iterations
- **Activation Function**: Sigmoid or Hyperbolic Tangent (radio buttons)

#### Fixed Parameters
- **Input Features**: 5 (all bird characteristics)
- **Output Classes**: 3 (species A, B, C)
- **Weight Initialization**: Small random numbers

### GUI Layout
```
┌─────────────────────────────────────┐
│ Network Architecture Configuration  │
├─────────────────────────────────────┤
│ Hidden Layers: [2] ▼                │
│ Layer 1 Neurons: [10] ▼             │
│ Layer 2 Neurons: [8] ▼              │
│ Learning Rate: [0.01] ___           │
│ Epochs: [1000] ____                 │
│ ☑ Use Bias                          │
│ ○ Sigmoid ● Tanh                    │
├─────────────────────────────────────┤
│ [Train Network] [Test Sample]       │
├─────────────────────────────────────┤
│ Results: Accuracy, Confusion Matrix │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
Task2-Backpropagation-MLP/
├── src/
│   ├── MLP.py                 # MLP class implementation
│   ├── utils.py               # Data loading and preprocessing
│   ├── app.py                 # Main GUI application
│   └── task2-notebook.py     
├── data/
│   └── birds.csv              # Bird species dataset
├── report/
│   └── mlp_report.pdf          
└── README.md
```

## 🚀 Implementation Details

### MLP Class Structure
```python
class MLP:
    def __init__(self, layers, activation='sigmoid', use_bias=True):
        self.layers = layers  # [5, 10, 8, 3] example
        self.activation = activation
        self.use_bias = use_bias
        self.weights = []
        self.biases = []
        
    def forward_pass(self, X):
        # Compute activations layer by layer
        
    def backward_pass(self, X, y):
        # Compute gradients using backpropagation
        
    def update_weights(self, learning_rate):
        # Apply gradient descent updates
        
    def train(self, X, y, epochs, learning_rate):
        # Training loop with error monitoring
```

### Activation Function Implementations
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

## 📈 Performance Analysis

### Experimental Setup
Test various configurations to find optimal parameters:

#### Network Architectures to Test
1. **Single Hidden Layer**: [5, n, 3] where n ∈ {5, 10, 15, 20}
2. **Two Hidden Layers**: [5, n1, n2, 3] various combinations
3. **Three Hidden Layers**: [5, n1, n2, n3, 3] selected configurations

#### Hyperparameter Grid Search
- **Learning Rates**: [0.001, 0.01, 0.1, 0.5]
- **Epochs**: [500, 1000, 2000, 5000]
- **Activation Functions**: [Sigmoid, Tanh]
- **Bias**: [With Bias, Without Bias]

## 🔍 Key Features

### Algorithm Implementation
- ✅ **Full Backpropagation**: Forward and backward pass
- ✅ **Flexible Architecture**: Variable hidden layers and neurons
- ✅ **Multiple Activations**: Sigmoid and Hyperbolic Tangent
- ✅ **Bias Support**: Optional bias terms
- ✅ **Weight Initialization**: Proper random initialization

### Evaluation Metrics
- ✅ **Custom Confusion Matrix**: Implemented from scratch
- ✅ **Accuracy Calculation**: Overall and per-class accuracy
- ✅ **Training Progress**: Loss and accuracy tracking
- ✅ **Convergence Analysis**: Training curve visualization

## 📊 Deliverables

### Code Requirements
- **Complete MLP Implementation**: All algorithms from scratch
- **Interactive GUI**: User-friendly parameter configuration
- **Comprehensive Testing**: Multiple architecture experiments
- **Performance Analysis**: Detailed comparison of configurations

## 🎯 Learning Objectives

This project teaches:
- **Deep Learning Fundamentals**: Multi-layer neural networks
- **Backpropagation Algorithm**: Gradient-based learning
- **Network Design**: Architecture choices and their impact
- **Hyperparameter Tuning**: Systematic parameter optimization
- **Performance Evaluation**: Comprehensive model assessment
- **Software Engineering**: Clean, modular code organization

## ⚡ Quick Start

### Installation
```bash
pip install numpy matplotlib pandas streamlit
```

### Running the Application
```bash
streamlit run src/app.py
```

### Usage Workflow
1. **Configure Network**: Set hidden layers and neurons
2. **Set Parameters**: Choose learning rate, epochs, activation
3. **Train Model**: Click train and monitor progress
4. **Evaluate Results**: Review accuracy and confusion matrix
5. **Experiment**: Try different configurations
6. **Document Best**: Record optimal parameters and performance

## 🏆 Success Metrics

- ✅ **Functional MLP**: Proper backpropagation implementation
- ✅ **GUI Completeness**: All required features working
- ✅ **High Accuracy**: >90% testing accuracy achieved
- ✅ **Thorough Analysis**: Comprehensive parameter comparison
- ✅ **Clean Code**: Well-structured, documented implementation
- ✅ **Detailed Report**: Professional analysis with screenshots

---

**Goal**: Master the fundamentals of multi-layer neural networks and understand how architectural choices affect learning performance in multi-class classification tasks.