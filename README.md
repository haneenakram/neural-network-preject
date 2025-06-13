# Fine-Grained Fruit Quality Assessment using Deep Learning

## Team CHP_5

- **Haneen Akram Ahmed**
- **Menna Ali Thabet**
- **Mohamed Ashraf Fathy**
- **Reem Ahmed Ismail**
- **Zeina Shawkat**
- **Islam Hesham**

## 🚀 Neural Networks Course - Final Project

This repository showcases an advanced deep learning project for **fine-grained fruit quality assessment**, implementing and comparing **6 different neural network architectures** to classify fruits into specific ripeness and quality categories. This project demonstrates mastery of modern deep learning techniques, from CNNs to Vision Transformers.

> **💡 Course Foundation**: This project builds upon fundamental neural network concepts learned through implementing [Perceptron & Adaline algorithms](../../tree/task1) and [Multi-Layer Perceptron with Backpropagation](../../tree/task2) from scratch.

## 📋 Project Overview

### 🎯 Objective

Develop robust deep learning models capable of performing fine-grained classification of fruit quality using image features, specifically targeting:

- **Banana categories**: overripe, ripe, rotten, unripe
- **Tomato categories**: fully ripened, green, half ripened

### 🏆 Key Achievements

- **96.00% validation accuracy** using CNN + Transformer Hybrid
- **6 different architectures** implemented and compared
- **Severe class imbalance handled** (39.74:1 ratio)
- **7,395 images** processed across 7 quality categories
- **Custom Vision Transformer** designed specifically for fruit textures

## 📊 Dataset Challenge

The dataset presents a significant **class imbalance challenge**:

| Category             | Count | Percentage |
| -------------------- | ----- | ---------- |
| banana_rotten        | 1,987 | 26.9%      |
| banana_ripe          | 1,440 | 19.5%      |
| banana_overripe      | 1,395 | 18.9%      |
| banana_unripe        | 1,370 | 18.5%      |
| tomato_green         | 334   | 4.5%       |
| tomato_half_ripened  | 81    | 1.1%       |
| tomato_fully_ripened | 50    | 0.7%       |

**Challenge**: 39.74:1 imbalance ratio between largest and smallest classes

## 🏗️ Model Architectures & Results

### 🥇 Top Performing Models

| Model              | Validation Accuracy | Key Innovation |
| ------------------ | ------------------- | -------------- |
| **CNN + Transformer** | **96.00%** 🏆 | Hybrid local-global feature fusion |
| **CoAtNet**            | **95.45%** 🥈 | Convolutional + Attention integration |
| **Vision Transformer** | **95.39%** 🥉 | Custom ViT for fruit texture analysis |
| ResNet-34             | ~95%        | Auxiliary classifier enhancement |
| Custom ResNet-50      | 93-94%      | Enhanced regularization |
| Baseline CNN          | 92.2%       | Three-block foundation |

### 🧠 Architecture Highlights

#### 1. Vision Transformer (Custom Design)
```
Input: 224×224×3 → 784 patches (8×8)
├── Patch Embedding: 128-dim + positional encoding
├── Transformer: 8 layers, 8 heads, GELU activation  
├── Global Average Pooling
└── MLP Head: 2048→1024→7 classes
```
- **Parameters**: 7.77M (29.65 MB)
- **Innovation**: Optimized patch size for fruit texture details

#### 2. CNN + Transformer Hybrid 🏆
- **Best Performance**: 96.00% validation accuracy
- **Design**: CNN feature extraction + Transformer attention
- **Advantage**: Combines local texture analysis with global reasoning

#### 3. CoAtNet (Convolution + Attention)
- **Architecture**: Conv stem → MBConv blocks → Transformer blocks
- **Strength**: Seamless integration of convolutional and attention mechanisms

## 🔧 Technical Implementation

### Advanced Training Techniques
- **Class Weighting**: Dynamic weights (0.48 to 19.02) for imbalance handling
- **Data Augmentation**: Rotation, zoom, shifts, brightness adjustment
- **Optimization**: AdamW with weight decay (1e-5)
- **Learning Rate**: 1e-4 with reduction on plateau
- **Regularization**: Early stopping, dropout, batch normalization

### Key Technical Skills Demonstrated
- ✅ **Custom Architecture Design**: Built Vision Transformer from scratch
- ✅ **Class Imbalance Mastery**: Effective weighting strategies
- ✅ **Hybrid Model Development**: CNN-Transformer fusion
- ✅ **Performance Optimization**: Systematic hyperparameter tuning
- ✅ **Deep Learning Engineering**: Complete training pipelines

## 📁 Repository Structure

```
fruit-quality-assessment/
├── CNN/                              # Baseline convolutional model
│   ├── CNN_Report.pdf
│   └── cnn.ipynb
├── CNN + Transformer Hybrid Model/   # Best performing model
│   ├── CNN+Transformer.ipynb
│   └── CNN+Transformer_Hybrid_Report.pdf
├── CoAtNet/                          # Convolutional + Attention
│   ├── coAtNet.ipynb
│   └── CoAtNet_Report.pdf
├── ResNet-34/                        # Residual network variant
│   ├── ResNet34.ipynb
│   └── ResNet34_Report.pdf
├── ResNet-50/                        # Custom ResNet implementation
│   ├── resnet50.ipynb
│   └── ResNet50_Report.pdf
├── ViT/                             # Custom Vision Transformer
│   ├── ViT.ipynb
│   ├── Vision_Transformer_Report.py
│   ├── vit_classes.py
│   ├── fruit_quality_vit_complete.keras
│   └── vit_weights.h5
├── class_indeces.json
└── README.md
```
## 📈 Course Learning Journey

This project represents the culmination of a comprehensive neural networks course:

### 🎓 Foundation → Advanced Progression

```
├── Task 1: Perceptron & Adaline (Branch: task1-perceptron-adaline)
│   ├── Single-layer networks from scratch
│   ├── Binary classification fundamentals  
│   ├── Interactive GUI development
│   └── Custom evaluation metrics
│
├── Task 2: Multi-Layer Perceptron (Branch: task2-backpropagation-mlp)
│   ├── Backpropagation algorithm implementation
│   ├── Multi-class classification
│   ├── Interactive GUI development
│   ├── Architecture design principles
│   └── Hyperparameter optimization
│
└── Final Project: Advanced Deep Learning (Main Branch)
    ├── State-of-the-art architectures
    ├── Vision Transformers & Hybrid models
    ├── Real-world dataset challenges
    └── Production-ready performance
```

### 🔗 Related Course Work
- **[Task 1: Perceptron & Adaline](../../tree/task1)** - Foundation algorithms
- **[Task 2: Backpropagation MLP](../../tree/task2)** - Core deep learning concepts

## 🎯 Key Learning Outcomes

### **Deep Learning Expertise**
- Designed and implemented Vision Transformers from mathematical foundations
- Built hybrid CNN-Transformer architectures for optimal performance
- Mastered attention mechanisms and their application to computer vision
- Achieved state-of-the-art results on challenging imbalanced datasets

### **ML Engineering Skills**
- Handled severe class imbalance using advanced weighting techniques
- Implemented comprehensive model comparison frameworks
- Applied systematic hyperparameter optimization strategies
- Developed robust training pipelines with proper regularization

### **Research & Analysis**
- Conducted thorough experimental comparisons across 6 architectures
- Performed ablation studies on architectural components
- Generated detailed technical reports with performance analysis
- Demonstrated scientific rigor in model evaluation and selection

## 🏆 Impact & Results

- **🎯 Accuracy**: 96% validation accuracy on challenging dataset
- **⚖️ Balance**: Successfully handled 39:1 class imbalance
- **🧠 Innovation**: Custom ViT architecture outperformed standard approaches
- **📊 Comparison**: Systematic evaluation of 6 different deep learning architectures
- **📈 Scalability**: Models ready for production deployment

## 📚 Technical References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Dai, Z., et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." NeurIPS 2021.
3. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

---

**💡 This project demonstrates advanced deep learning expertise, from implementing algorithms from scratch to designing state-of-the-art architectures for real-world applications.**

_Neural Networks Course - Spring 2025_