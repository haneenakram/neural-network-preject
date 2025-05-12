# the Fine-Grained Fruit Quality Assessment project solutions
---

## 1. CNN-Based Custom Architecture (Baseline Model) => Menna

**Goal**: Create a simple yet effective CNN from basic layers to classify fruit quality.

**Architecture**:
-	Conv2D → ReLU → MaxPooling → Dropout
-	Repeat 3–4 blocks
-	Flatten → Dense → Output Layer (Softmax)

**Features**:
-	Fully explainable architecture
-	No use of Keras built-in applications
-	Suitable for smaller datasets

**Advantages**:
-	Easy to interpret
-	Good as a baseline

---

## 2. Transfer Learning with Pretrained Weights (Without using keras.applications) => Zeina

**Goal**: Use pretrained weights but implement architecture manually to comply with the rule.

**Example**: Manually implement ResNet-34 or EfficientNet using TensorFlow/Keras layers.

**Steps**:
-	Re-implement architecture like ResNet/EfficientNet using custom code.
-	Load pretrained weights (if allowed), or train from scratch on your dataset.
-	Add custom classifier layers at the end.

**Advantages**:
-	Leverages the power of proven architectures
-	Better performance than a small CNN

**Tip**: Use repositories like timm or load weights using load_weights for custom model structure.

---

## 3. Vision Transformer (ViT or Swin Transformer) => Haneen

**Goal**: Use attention-based architecture for fine-grained image classification.

**Options**:
-	Implement a Vision Transformer (ViT) manually using:
-	PatchEmbedding
-	TransformerEncoder blocks (Multi-head self-attention + MLP + LayerNorm)
-	Or use Swin Transformer with shifted windows.

**Challenges**:
-	Requires more compute
-	Needs careful regularization

**Advantages**:
-	Captures global features
-	Especially effective for subtle visual differences (ripe vs overripe)

---

## 4. Handling Class Imbalance (Essential for All Models) => zeina

Since the dataset is unbalanced, apply at least one of these techniques:
- Weighted loss (class_weight in Keras, or manual weighting in loss function)
-	Data Augmentation for minority classes (e.g., rotation, brightness, flips)
-	Oversampling using ImageDataGenerator or custom generator
-	Focal Loss: Helps focus learning on harder, less frequent samples

---

## 5. Ensemble Learning (Optional Final Approach) => haneen

**Goal**: Combine the predictions of the three diverse models.

**Methods**:
-	Majority Voting
-	Averaging Softmax Scores

**When to use:**
-	If individual models perform well but have different strengths

---

## Data Preparation Steps (Must Include) => zeina
-	Resize images consistently (e.g., 224x224)
-	Normalize images
-	Label encode the categories
-	Use ImageDataGenerator or tf.data pipeline
-	Data split: Train / Validation / Test (stratified if possible)

---

## Model Training & Evaluation => menna
-	Use EarlyStopping, ModelCheckpoint
-	Save weights (model.save_weights('model_weights.keras'))
-	Evaluate on validation set using accuracy, confusion matrix
-	Export predictions to CSV for Kaggle

---

## Final Deliverables
-	Three distinct models (CNN, reimplemented ResNet, Transformer)
-	Detailed report:
-	Architecture diagrams
-	Training logs or screenshots
-	Loss/accuracy plots
-	Kaggle scores
-	Analysis of what worked best and why
