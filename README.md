# Convolutional Neural Network for ECG Heartbeat Classification

A from-scratch implementation of a Convolutional Neural Network for automated classification of ECG heartbeat signals, demonstrating deep understanding of neural network architectures, backpropagation algorithms, and cardiovascular signal processing.

## Project Overview

This project implements a complete CNN architecture without relying on high-level frameworks like TensorFlow or PyTorch. The system classifies ECG heartbeat signals into 5 categories using the MIT-BIH Arrhythmia Database, showcasing fundamental understanding of neural network mathematics and implementation details.

## Mathematical Foundation

### Convolutional Neural Network Architecture

The network follows a classical CNN structure:
```
Input (187 features) → Convolution → ReLU → Max Pooling → Flatten → FC Layer → ReLU → Output Layer → Softmax
```

### Forward Propagation Mathematics

**1. Convolution Operation:**
```
z_conv[f,i] = Σ(j=0 to k-1) x[i+j] * W_conv[f,j] + b_conv[f]
```
Where:
- `f` = filter index (0 to conv_filters-1)
- `i` = output position (0 to input_length-kernel_size)
- `k` = kernel_size
- Output size = input_length - kernel_size + 1

**2. ReLU Activation:**
```
a_relu[i] = max(0, z_conv[i])
```

**3. Max Pooling:**
```
pooled[f,k] = max(a_relu[f, k*pool_size : (k+1)*pool_size])
```
Reduces spatial dimensions by factor of `pool_size`

**4. Fully Connected Layers:**
```
z_fc1 = flattened · W_fc1 + b_fc1
a_fc1 = ReLU(z_fc1)
z_fc2 = a_fc1 · W_fc2 + b_fc2
```

**5. Softmax Output:**
```
y_hat[i] = exp(z_fc2[i]) / Σ(j=0 to num_classes-1) exp(z_fc2[j])
```

### Backpropagation Implementation

**Output Layer Gradient:**
```
∂L/∂z_fc2 = y_hat - y_true
```

**Hidden Layer Gradients:**
```
∂L/∂W_fc2 = a_fc1^T · ∂L/∂z_fc2
∂L/∂b_fc2 = ∂L/∂z_fc2
∂L/∂a_fc1 = W_fc2 · ∂L/∂z_fc2
```

**ReLU Derivative:**
```
∂L/∂z_fc1 = ∂L/∂a_fc1 ⊙ ReLU'(z_fc1)
where ReLU'(x) = 1 if x > 0, else 0
```

**Max Pooling Backpropagation:**
The gradient flows only through the maximum element in each pooling window:
```
∂L/∂a_relu[f,i] = ∂L/∂pooled[f,k] if i == argmax(pooling_window), else 0
```

**Convolution Layer Gradients:**
```
∂L/∂W_conv[f,j] = Σ(i=0 to output_size-1) ∂L/∂z_conv[f,i] · x[i+j]
∂L/∂b_conv[f] = Σ(i=0 to output_size-1) ∂L/∂z_conv[f,i]
```

## Technical Implementation

### Model Architecture (`model.py`)

**CNN Class Structure:**
- **Input Layer**: 187-dimensional ECG signal vectors
- **Convolutional Layer**: 12 filters with kernel size 5
- **Max Pooling**: Pool size 2 with stride 2
- **Fully Connected Layer**: 64 hidden units with ReLU activation
- **Output Layer**: 5 classes with softmax activation

**Key Implementation Features:**
- Manual convolution operation without external libraries
- Proper gradient computation for all layer types
- Gradient clipping to prevent exploding gradients
- Numerical stability in softmax computation

### Data Processing Pipeline

**Signal Preprocessing:**
```python
# Z-score normalization per signal
X_normalized = (X - μ) / (σ + ε)
```
Where ε = 1e-6 prevents division by zero

**Label Encoding:**
- One-hot encoding for categorical cross-entropy loss
- 5 classes representing different heartbeat types

### Training Process (`train.py`)

**Hyperparameter Configuration:**
- Learning rate: 0.005 (optimized for stability)
- Batch size: 32 (balance between efficiency and gradient noise)
- Epochs: 20 (sufficient for convergence)
- Gradient clipping: 1.0 (prevents exploding gradients)

**Training Loop Implementation:**
1. **Data Shuffling**: Random permutation each epoch
2. **Mini-batch Processing**: Efficient memory usage
3. **Forward Pass**: Signal → Features → Predictions
4. **Backward Pass**: Error → Gradients → Parameter Updates
5. **Model Persistence**: Save trained weights via joblib

### Evaluation Framework (`evaluate.py`)

**Comprehensive Metrics:**
- **Accuracy**: Overall classification correctness
- **Precision**: Class-specific prediction quality (weighted average)
- **Recall**: Class-specific detection completeness (weighted average)
- **F1 Score**: Harmonic mean of precision and recall

**Visualization Components:**
1. **Performance Bar Chart**: Quick metric comparison
2. **Confusion Matrix Heatmap**: Detailed classification analysis

## Signal Processing Considerations

### ECG Signal Characteristics
- **Temporal Resolution**: 187 sample points per heartbeat
- **Morphological Features**: P-QRS-T wave complexes
- **Noise Handling**: Z-score normalization removes baseline drift
- **Arrhythmia Detection**: 5-class classification system

### Feature Extraction via Convolution
- **Local Pattern Detection**: 5-point kernels capture QRS morphology
- **Translation Invariance**: Convolution detects patterns regardless of position
- **Hierarchical Learning**: Multiple filters learn diverse cardiac features
- **Dimensionality Reduction**: Max pooling reduces computational complexity

## Performance Analysis

### Classification Metrics Interpretation

**Accuracy**: Overall system reliability across all heartbeat types
**Precision**: Minimizes false positive diagnoses (crucial for medical applications)
**Recall**: Minimizes missed arrhythmias (critical for patient safety)
**F1 Score**: Balanced measure accounting for class imbalance

### Confusion Matrix Analysis
The confusion matrix reveals:
- **Diagonal Elements**: Correct classifications per class
- **Off-diagonal Elements**: Specific misclassification patterns
- **Class Imbalance**: Some arrhythmia types may be underrepresented

## Advanced Implementation Details

### Gradient Computation Accuracy
The backpropagation implementation manually computes all partial derivatives:
- **Chain Rule Application**: Proper gradient flow through all layers
- **Jacobian Calculations**: Correct handling of multi-dimensional operations
- **Memory Efficiency**: Minimal storage of intermediate values

### Numerical Stability Features
- **Softmax Overflow Prevention**: Subtract max value before exponential
- **Gradient Clipping**: Prevent exploding gradients during training
- **Small Weight Initialization**: Prevent vanishing gradients
- **Epsilon in Normalization**: Avoid division by zero

### Architectural Decisions

**Filter Count (12)**: Balance between feature diversity and computational cost
**Kernel Size (5)**: Optimal for capturing QRS complex features
**Pool Size (2)**: Standard downsampling without excessive information loss
**Hidden Units (64)**: Sufficient capacity for pattern integration

## Medical Domain Relevance

### Clinical Applications
- **Automated ECG Screening**: Reduce cardiologist workload
- **Real-time Monitoring**: Continuous arrhythmia detection
- **Telemedicine**: Remote cardiac monitoring systems
- **Research Tool**: Large-scale cardiac rhythm analysis

### Regulatory Considerations
- **FDA Compliance**: Algorithm transparency for medical device approval
- **Interpretability**: Understanding model decision-making process
- **Validation**: Rigorous testing on diverse patient populations
- **Safety**: High recall critical to avoid missing dangerous arrhythmias

## Key Achievements

### Technical Mastery
- **From-scratch Implementation**: Complete neural network without frameworks
- **Mathematical Rigor**: Proper gradient computation and optimization
- **Signal Processing**: Domain-appropriate preprocessing techniques
- **Performance Evaluation**: Comprehensive metric analysis

### Software Engineering
- **Modular Design**: Separate training, model, and evaluation components
- **Code Documentation**: Clear mathematical explanations
- **Reproducibility**: Consistent results through proper initialization
- **Scalability**: Efficient batch processing implementation

## Model Performance Results

### Achieved Metrics

The CNN model achieved the following performance on the MIT-BIH test dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | **91.63%** |
| **Precision** | **88.85%** |
| **Recall** | **91.63%** |
| **F1 Score** | **89.59%** |

### Performance Analysis

**Strong Overall Performance**: The model demonstrates robust classification capability with over 91% accuracy, indicating reliable heartbeat pattern recognition across the 5 ECG classes.

**Balanced Precision-Recall Trade-off**: The precision (88.85%) and recall (91.63%) scores show a well-balanced model that avoids both excessive false positives and missed detections - crucial for medical applications.

**Clinical Relevance**: The 91.63% recall indicates that the model successfully identifies most arrhythmic events, which is critical for patient safety in cardiac monitoring applications.

**F1 Score Interpretation**: The F1 score of 89.59% demonstrates consistent performance across different heartbeat classes, suggesting the model generalizes well despite potential class imbalances in the dataset.

## Implementation Limitations and Constraints

### Computational Challenges

**Extended Training Time**: Training sessions often exceeded 6 hours due to the from-scratch implementation without optimized libraries like cuDNN or GPU acceleration. This significantly limited iterative development and hyperparameter tuning opportunities.

**Limited Hyperparameter Exploration**: The computational overhead prevented extensive experimentation with:
- Different architectural configurations (filter counts, kernel sizes)
- Advanced optimization algorithms (Adam, RMSprop)
- Learning rate scheduling strategies
- Regularization techniques (dropout, batch normalization)

**Resource Constraints**: Without access to high-performance computing resources or GPU acceleration, the development cycle was constrained to minimal experimentation, potentially leaving performance optimizations unexplored.

### From-Scratch Implementation Trade-offs

**Optimization Limitations**:
- Manual implementation lacks highly optimized matrix operations found in production frameworks
- No automatic mixed precision training or memory optimization
- Limited to basic SGD optimizer without momentum or adaptive learning rates

**Development Efficiency**:
- Extensive debugging time required for gradient computation validation
- Manual implementation of standard operations that are heavily optimized in frameworks
- Increased development complexity compared to high-level API usage

**Scalability Constraints**:
- Memory usage not optimized for larger batch sizes
- No distributed training capabilities for larger datasets
- Limited ability to experiment with deeper architectures due to computational cost

### Potential Performance Improvements

**With Additional Resources**:
- **GPU Acceleration**: Could reduce training time from 6+ hours to minutes, enabling extensive hyperparameter search
- **Advanced Optimizers**: Adam or RMSprop could improve convergence and final performance
- **Regularization Techniques**: Dropout and batch normalization could enhance generalization
- **Architecture Search**: Systematic exploration of different CNN configurations
- **Data Augmentation**: Synthetic ECG variations could improve robustness

**Framework Migration Benefits**:
- Access to pre-trained cardiac models for transfer learning
- Automatic differentiation reducing implementation errors
- Advanced optimization techniques and learning rate schedules
- Production-ready deployment capabilities

### Research and Development Insights

Despite computational limitations, this implementation demonstrates:

**Fundamental Understanding**: Achieving 91%+ accuracy with a basic from-scratch implementation validates deep comprehension of CNN principles and medical signal processing.

**Algorithm Robustness**: The strong performance suggests that the core architectural choices and mathematical implementation are sound, providing a solid foundation for future enhancements.

**Practical Constraints**: The development experience highlights the importance of computational resources in deep learning research and the value of optimized frameworks for iterative improvement.

**Educational Value**: Building from scratch provided invaluable insights into gradient flow, numerical stability, and the intricacies of neural network training that are often abstracted away in high-level frameworks.

This project demonstrates my understanding of neural network fundamentals, medical signal processing, and the intersection of machine learning with healthcare applications. While computational constraints limited optimization opportunities, the implementation showcases both theoretical knowledge and practical programming skills essential for developing reliable AI systems in critical domains, achieving clinically relevant performance despite resource limitations.
