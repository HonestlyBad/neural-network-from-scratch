# Neural Network from Scratch

A simple neural network implementation in Python/NumPy for classifying handwritten digits from the MNIST dataset.

## Network Architecture

This is a 2-layer neural network with:
- **Input layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation (one per digit 0-9)

## Mathematical Foundation

### 1. Parameter Initialization

Weights and biases are initialized randomly in the range $[-0.5, 0.5]$:

$$W^{[1]} \in \mathbb{R}^{10 \times 784}, \quad b^{[1]} \in \mathbb{R}^{10 \times 1}$$

$$W^{[2]} \in \mathbb{R}^{10 \times 10}, \quad b^{[2]} \in \mathbb{R}^{10 \times 1}$$

### 2. Forward Propagation

#### Layer 1 (Hidden Layer)

Linear transformation:
$$Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$$

ReLU activation:
$$A^{[1]} = \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$$

#### Layer 2 (Output Layer)

Linear transformation:
$$Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$$

Softmax activation (converts to probabilities):
$$A^{[2]}_i = \text{softmax}(Z^{[2]})_i = \frac{e^{Z^{[2]}_i}}{\sum_{j=1}^{10} e^{Z^{[2]}_j}}$$

### 3. One-Hot Encoding

Labels $Y$ are converted to one-hot vectors. For a label $y = 3$:

$$\hat{Y} = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T$$

### 4. Backpropagation

#### Output Layer Gradients

Error at output layer (derivative of cross-entropy loss with softmax):
$$dZ^{[2]} = A^{[2]} - \hat{Y}$$

Weight gradient:
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} \cdot (A^{[1]})^T$$

Bias gradient:
$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

#### Hidden Layer Gradients

Error propagated back through weights:
$$dZ^{[1]} = (W^{[2]})^T \cdot dZ^{[2]} \odot \text{ReLU}'(Z^{[1]})$$

Where ReLU derivative is:
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$

Weight gradient:
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} \cdot X^T$$

Bias gradient:
$$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

### 5. Gradient Descent Update

Parameters are updated using learning rate $\alpha$:

$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$

$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

### 6. Prediction & Accuracy

Prediction is the class with highest probability:
$$\hat{y} = \arg\max(A^{[2]})$$

Accuracy:
$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}(\hat{y}_i = y_i)$$

## Usage

```python
python main.py
```

The script trains on the MNIST dataset for 500 iterations with a learning rate of 0.1.

## Data Preprocessing

- Pixel values are normalized to $[0, 1]$ by dividing by 255
- Data is split into training (59,000 samples) and development (1,000 samples) sets