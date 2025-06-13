# Fine-Grained Fruit Quality Assessment using Deep Learning

## Team CHP_5

- **Haneen Akram Ahmed**
- **Menna Ali Thabet**
- **Mohamed Ashraf Fathy**
- **Reem Ahmed Ismail**
- **Zeina Shawkat**
- **Islam Hesham**

## 📋 Project Overview

This project implements multiple deep learning architectures for fine-grained fruit quality assessment, focusing on classifying bananas and tomatoes into specific ripeness and quality categories. The models are designed to automatically assess and categorize fruit quality based on visual characteristics, addressing the challenge of class imbalance in agricultural datasets.

### 🎯 Objective

Develop robust deep learning models capable of performing fine-grained classification of fruit quality using image features, specifically targeting:

- **Banana categories**: overripe, ripe, rotten, unripe
- **Tomato categories**: fully ripened, green, half ripened

## 📊 Dataset

The dataset contains **7,395 images** across 7 quality categories with significant class imbalance:

| Category             | Count | Percentage |
| -------------------- | ----- | ---------- |
| banana_rotten        | 1,987 | 26.9%      |
| banana_ripe          | 1,440 | 19.5%      |
| banana_overripe      | 1,395 | 18.9%      |
| banana_unripe        | 1,370 | 18.5%      |
| tomato_green         | 334   | 4.5%       |
| tomato_half_ripened  | 81    | 1.1%       |
| tomato_fully_ripened | 50    | 0.7%       |

**Imbalance Ratio**: 39.74 (largest/smallest class)

## 🏗️ Model Architectures

We implemented and compared **6 different architectures** to solve this fine-grained classification problem:

### 1. Vision Transformer (ViT) - **Best Performance** 🏆

- **Custom ViT architecture** designed specifically for fruit quality assessment
- **Validation Accuracy**: 95.39%
- **Parameters**: 7.77M (29.65 MB)

**Key Components**:

- Patch Creation: 224×224×3 → 784 patches (8×8)
- Embedding: 128-dimensional patch embeddings + positional encoding
- Transformer: 8 layers, 8 attention heads, GELU activation
- Classification Head: Global average pooling + MLP (2048→1024→7)

### 2. CNN + Transformer Hybrid

- **Validation Accuracy**: 96.00%
- **Parameters**: ~7.5M
- Combines CNN feature extraction with transformer attention mechanism

### 3. CoAtNet (Convolutional + Attention)

- **Validation Accuracy**: 95.45%
- Hybrid architecture combining:
  - Convolutional stem for local features
  - MBConv blocks for efficient representation
  - Transformer blocks for global reasoning

### 4. Baseline CNN

- **Validation Accuracy**: 92.2%
- Three convolutional blocks (32, 64, 128 filters)
- ReLU activations, max pooling, dropout regularization

### 5. ResNet-34 with Auxiliary Classifier

- **Validation Accuracy**: ~95%
- Standard ResNet34 with additional auxiliary output
- Cosine learning rate decay with Adam optimizer

### 6. Custom ResNet-50

- **Validation Accuracy**: 93-94%
- Modified ResNet50 with enhanced regularization
- L2 regularization, dropout, batch normalization

## 🔧 Training Configuration

### Data Preprocessing & Augmentation

- **Image Size**: 224×224×3
- **Normalization**: Pixel values scaled to [0,1]
- **Augmentation Techniques**:
  - Rotation (±20-40°)
  - Zoom (±10-20%)
  - Width/height shifts (±10-20%)
  - Shear transformation (10-20%)
  - Horizontal flipping
  - Brightness/contrast adjustment

### Training Strategy

- **Batch Size**: 32
- **Epochs**: 50-60 (with early stopping)
- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning Rate**: 1e-4 with reduction on plateau
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Class Weighting**: Dynamic weights to handle imbalance (0.48 to 19.02)

## 📈 Results Summary

| Model              | Validation Accuracy | Training Accuracy | Parameters |
| ------------------ | ------------------- | ----------------- | ---------- |
| Vision Transformer | 95.39%              | 96.25%            | 7.77M      |
| CNN + Transformer  | 96.00%              | 97.00%            | 7.5M       |
| CoAtNet            | 95.45%              | 96.67%            | -          |
| ResNet-34          | ~95%                | 92.76%            | -          |
| Custom ResNet-50   | 93-94%              | 97.42%            | -          |
| Baseline CNN       | 92.2%               | 94.2%             | -          |

### Key Findings

- **Custom architectures outperformed pre-trained models** for this specific task
- **Class weighting proved more effective** than synthetic oversampling
- **Transformer-based models showed superior performance** in handling fine-grained distinctions
- **Consistent performance** across both banana and tomato categories despite severe imbalance

## 🛠️ Implementation Details

### Technologies Used

- **Framework**: TensorFlow/Keras
- **Environment**: Kaggle Notebooks
- **Libraries**: NumPy, Pandas, Matplotlib, scikit-learn
- **Hardware**: GPU acceleration for training

### Training Techniques

- **Early Stopping**: Patience of 10 epochs
- **Model Checkpointing**: Save best weights
- **Learning Rate Scheduling**: Reduction on plateau
- **Mixed Precision Training**: For computational efficiency
- **Stratified Validation Split**: 80/20 train/validation

## 📁 Repository Structure

```
fruit-quality-assessment/
├── CNN/
│   ├── CNN_Report.pdf
│   ├── cnn.ipynb
│   ├── model_architecture.py
│   └── training_logs/
├── CNN + Transformer Hybrid Model/
│   ├── CNN+Transformer.ipynb
│   └── CNN+Transformer_Hybrid_Report.pdf
├── CoAtNet/
│   ├── coAtNet.ipynb
│   └── CoAtNet_Report.pdf
├── ResNet-34/
│   ├── ResNet34.ipynb
│   └── ResNet34_Report.pdf
├── ResNet-50/
│   ├── resnet50.ipynb
│   ├── augmentation_pipeline.py
│   └── ResNet50_Report.pdf
├── ViT/
│   ├── ViT.ipynb
│   ├── Vision_Transformer_Report.py
│   ├── vit_classes.py
│   ├── fruit_quality_vit_complete.keras
│   └── vit_weights.h5
├── .gitignore
└── README.md
```

## 🔍 Key Insights

### What Worked Best

1. **Custom Vision Transformer** architecture specifically designed for the task
2. **Class weighting** over synthetic data augmentation for imbalance handling
3. **Appropriate patch size** (8×8) for capturing fruit texture details
4. **Multi-stage learning rate scheduling** for optimal convergence

### Lessons Learned

- Pre-trained models don't always transfer well to specialized domains
- Fine-grained classification benefits from attention mechanisms
- Proper handling of class imbalance is crucial for fair evaluation
- Architectural choices should align with problem characteristics

## 🏆 Competition Results

- **Kaggle Competition**: Fine-Grained Fruit Quality Assessment
- **Best Submission**: Vision Transformer model with 95.39% validation accuracy
- **Submission Format**: CSV with ImageID, Class (0-6), and ClassName

## 📚 References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Dai, Z., et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." NeurIPS 2021.
3. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

## 📄 License

This project is part of an academic assignment for deep learning course. All rights reserved to the team members.

---

_Project completed as part of Neural Network course - Spring 2025_
