# Task 1: Perceptron & Adaline Learning Algorithms

## 🐦 Project Overview

This project implements two fundamental single-layer neural network algorithms for binary classification of bird species:
- **Perceptron Learning Algorithm**
- **Adaline (Adaptive Linear Neuron) Learning Algorithm**

Both algorithms are designed to classify bird species based on physical characteristics with an interactive GUI for parameter tuning and visualization.

## 📊 Dataset Description

### Birds Dataset
- **Total Samples**: 150 (50 samples per species)
- **Species**: A, B, C
- **Features**: 5 measurements per sample
  - `gender` (categorical - requires preprocessing)
  - `body_mass` (numerical)
  - `beak_length` (numerical)
  - `beak_depth` (numerical)
  - `fin_length` (numerical)

### Data Split Strategy
- **Training**: 30 randomly selected samples per class
- **Testing**: Remaining 20 samples per class
- **Binary Classification**: Select any 2 classes from {A, B, C}

## 🧠 Algorithms Implemented

### 1. Perceptron Learning Algorithm
- **Type**: Single-layer neural network
- **Activation**: Step function
- **Learning**: Error-correction based
- **Convergence**: Guaranteed for linearly separable data

**Algorithm Workflow**:
```
1. Initialize weights randomly
2. For each training sample:
   - Calculate output = step(w·x + b)
   - Update weights: w = w + η(target - output)x
   - Update bias: b = b + η(target - output)
3. Repeat until convergence or max epochs
```

### 2. Adaline Learning Algorithm
- **Type**: Adaptive linear neuron
- **Activation**: Linear activation with sigmoid squashing
- **Learning**: Mean Squared Error (MSE) minimization
- **Convergence**: Based on MSE threshold

**Algorithm Workflow**:
```
1. Initialize weights randomly
2. For each training sample:
   - Calculate linear output = w·x + b
   - Calculate error = target - output
   - Update weights: w = w + η·error·x
   - Update bias: b = b + η·error
3. Repeat until MSE < threshold or max epochs
```

## 🖥️ GUI Features

### User Input Parameters
- **Feature Selection**: Choose 2 out of 5 features
- **Class Selection**: Pick 2 classes (A&B, A&C, or B&C)
- **Learning Rate (η)**: Controls step size in learning
- **Number of Epochs (m)**: Maximum training iterations
- **MSE Threshold**: Convergence criterion for Adaline
- **Bias Option**: Include/exclude bias term
- **Algorithm Choice**: Perceptron or Adaline (radio buttons)

### Output Visualizations
- **Decision Boundary**: Visual line separating classes
- **Scatter Plot**: Data points colored by class
- **Confusion Matrix**: Classification performance metrics
- **Training Progress**: Loss/accuracy curves

## 📁 Project Structure

```
Task1-Perceptron-Adaline/
├── src/
│   ├── SLP.py                 # Perceptron algorithm implementation
│   ├── Adaline.py             # Adaline algorithm implementation
│   ├── backend.py             # Dataset loading and preprocessing
│   └── GUI.py                 # Main GUI application
├── data/
│   └── birds_dataset.csv      # Bird species dataset
├── report/
│   └── NN_report.pdf    # Detailed analysis and results
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install tkinter  # Usually comes with Python
```

### Running the Application
1. Clone the repository
2. Navigate to the project directory
3. Run the main GUI application:
```bash
python src/gui.py
```

### Usage Steps
1. **Load Dataset**: The birds dataset loads automatically
2. **Select Features**: Choose 2 features from dropdown menus
3. **Select Classes**: Pick 2 bird species to classify
4. **Set Parameters**: Configure learning rate, epochs, MSE threshold
5. **Choose Algorithm**: Select Perceptron or Adaline
6. **Train Model**: Click "Train" to start learning process
7. **View Results**: Examine decision boundary, confusion matrix, and accuracy
8. **Test Classification**: Input new samples for prediction

## 📈 Performance Analysis

### Feature Combinations Tested
All possible 2-feature combinations (10 total):
1. gender + body_mass
2. gender + beak_length  
3. gender + beak_depth
4. gender + fin_length
5. body_mass + beak_length
6. body_mass + beak_depth
7. body_mass + fin_length
8. beak_length + beak_depth
9. beak_length + fin_length
10. beak_depth + fin_length

### Class Combinations
- **Classes A & B**
- **Classes A & C** 
- **Classes B & C**

### Evaluation Metrics
- **Confusion Matrix** (implemented from scratch)
- **Overall Accuracy**
- **Decision Boundary Visualization**
- **Training Convergence Analysis**

## 🔍 Key Implementation Details

### Data Preprocessing
- **Gender Encoding**: Categorical variable converted to numerical
- **Feature Normalization**: Optional scaling for better convergence
- **Random Sampling**: Ensures fair train/test split

### Algorithm Differences
| Aspect | Perceptron | Adaline |
|--------|------------|---------|
| **Activation** | Step function | Linear + Sigmoid |
| **Learning Rule** | Error correction | Gradient descent |
| **Convergence** | Classification errors | MSE threshold |
| **Robustness** | Sensitive to outliers | More robust |

### Visualization Features
- **Interactive Plots**: Real-time decision boundary updates
- **Color Coding**: Different colors for each class
- **Grid Overlay**: Shows decision regions clearly
- **Confusion Matrix Heatmap**: Visual performance assessment

## 📊 Expected Results

### Analysis Report Contents
1. **Algorithm Comparison**: Perceptron vs Adaline performance
2. **Feature Importance**: Which features best discriminate classes
3. **Class Separability**: Linear separability analysis
4. **Parameter Sensitivity**: Effect of learning rate and epochs
5. **Best Configurations**: Optimal settings for each scenario

### Deliverables
- ✅ Complete source code (.py files)
- ✅ Birds dataset
- ✅ Visualization plots
- ✅ Comprehensive analysis report
- ✅ Performance metrics and confusion matrices

## 🎯 Learning Objectives

- Understand fundamental neural network concepts
- Implement learning algorithms from scratch
- Design intuitive user interfaces
- Analyze algorithm performance and limitations
- Practice data visualization and interpretation
- Write well-documented, maintainable code

## 📝 Notes and Constraints

- **No External ML Libraries**: Algorithms implemented from scratch
- **No Data Dropping**: Use complete dataset
- **Custom Confusion Matrix**: No sklearn.metrics allowed
- **Code Quality**: Emphasis on readable, maintainable code
- **Separation of Concerns**: Logic separated from UI

## 🏆 Success Criteria

- ✅ Both algorithms correctly implemented
- ✅ GUI functional with all required features
- ✅ Decision boundaries properly visualized
- ✅ Confusion matrices calculated accurately
- ✅ Comprehensive analysis report completed
- ✅ Code is well-documented and organized

---

**Remember**: Focus on code quality, thorough analysis, and clear visualizations. This project builds fundamental understanding of neural network learning principles!