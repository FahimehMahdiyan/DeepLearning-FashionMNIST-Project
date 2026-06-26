# Deep Learning Project

## Phase 1: Manual MLP Implementation & Gradient Analysis

### Overview
In this phase, we implement a Multi-Layer Perceptron (MLP) from scratch using only NumPy. The network is trained and evaluated on the **Fashion-MNIST** dataset.

This phase consists of the following tasks:
- **Task 1-1:** Scratch implementation of a fully connected neural network (MLP) with modular forward and backward passes.
- **Task 1-2:** Training, hyperparameter tuning, and evaluation on the Fashion-MNIST dataset.
- **Task 1-3:** Increasing network depth to observe and analyze the **Vanishing Gradient** phenomenon.
- **Task 1-4:** Implementing **He Initialization** to mitigate vanishing gradients and stabilize training.

---

### Requirements
The following Python libraries are required to run the code:
```bash
pip install numpy pandas matplotlib
```
---

### Dataset Setup
The project uses the Fashion-MNIST dataset. You will likely download 
a folder containing both .csv and .idx-ubyte files.

Note: For this implementation, we exclusively use the CSV versions, 
as they are straightforward to process with NumPy and Pandas.

1. Required Files:
   - Download the dataset (e.g., from Kaggle).
   - Ensure you have:
      - fashion-mnist_train.csv
      - fashion-mnist_test.csv

---

### Implementation Details

#### 🧠 Architecture
- **Input Layer:** 784 neurons (28x28 flattened pixels).
- **Hidden Layers:** ReLU activation (customizable depth and width).
- **Output Layer:** Softmax activation with Cross-Entropy Loss.
- **Optimization:** Stochastic Gradient Descent (SGD) with manual backpropagation.

#### 📈 Key Observations
- **Task 1-3 (Vanishing Gradient):** As we increased the depth to 3 layers using small normal initialization, we observed that the gradient norm in the first layer was significantly smaller than the last layer, leading to slow or stalled learning.
- **Task 1-4 (He Initialization):** By using `sqrt(2/fan_in)`, we successfully balanced the variance of activations and gradients, allowing deeper networks to converge effectively.

### How to Run
> **Note:** Before running any phase-specific tasks, ensure that you have executed the **"Required Libraries"** cell at the top of the notebook to import all necessary dependencies.

1. Ensure the dataset CSV files are in the root directory.
2. Run the scripts sequentially or open the provided notebook.
3. The execution will output statistical results (Mean Accuracy ± STD) over 3 independent runs and plot the training curves.


***
***
***


## Phase 2: Optimization & Generalization

### Overview
In this phase, we focus on improving the performance and generalization ability of the neural network using modern deep learning techniques. Instead of implementing everything manually, we use **PyTorch** to experiment with different optimization and regularization strategies.

The experiments are performed on the **Fashion-MNIST** dataset using a 3-layer neural network architecture.

This phase consists of the following tasks:
- **Task 2-1:** Applying **Dropout** to reduce overfitting.
- **Task 2-2:** Using **Batch Normalization** to improve training stability and convergence speed.
- **Task 2-3:** Applying **L2 Regularization** and tuning the regularization parameter.
- **Task 2-4:** Performing **Bias–Variance Analysis** by varying the model complexity.

---

### Requirements
The following Python libraries are required to run the code:

```bash
pip install torch torchvision numpy matplotlib
```
---

### Dataset Setup
This phase uses the same **Fashion-MNIST** dataset as Phase 1.

1. Required Files:
   - Ensure the following files are available:
      - fashion-mnist_train.csv
      - fashion-mnist_test.csv

2. Data Preparation:
   - Normalize pixel values to the range **[0, 1]**.
   - Flatten each image into a **784-dimensional input vector**.

---

### Implementation Details

#### 🧠 Architecture
- **Input Layer:** 784 neurons (28×28 flattened pixels).
- **Hidden Layers:** 3 fully connected layers with **ReLU activation**.
- **Hidden Units:** 128 neurons per hidden layer.
- **Output Layer:** Softmax activation with **Cross-Entropy Loss**.
- **Optimizer:** Adam optimizer.
- **Learning Rate:** 0.001.

Different regularization and optimization techniques are applied on top of this base architecture.

#### 📈 Key Observations
- **Task 2-1 (Dropout):** By adding Dropout with probability **0.5** after each hidden layer, the model showed reduced overfitting and a smaller gap between training and validation loss.
- **Task 2-2 (Batch Normalization):** Adding Batch Normalization before ReLU activations improved training stability and allowed the network to converge faster, even with higher learning rates such as **0.01**.
- **Task 2-3 (L2 Regularization):** Random Search was used to tune the regularization coefficient **λ** from the set {0.00001, 0.0001, 0.001, 0.01, 0.1}. Moderate values of λ resulted in better generalization performance.
- **Task 2-4 (Bias–Variance Analysis):** By varying the hidden layer size across {32, 64, 128, 256, 512}, we observed that smaller networks tend to underfit (high bias), while larger networks tend to overfit (high variance).

---

### How to Run
> **Note:** Before running any phase-specific tasks, ensure that you have executed the **"Required Libraries"** cell at the top of the notebook to import all necessary dependencies.

1. Ensure the dataset CSV files are located in the project directory.
2. Run the Phase 2 training scripts or open the corresponding notebook.
3. The program will train the models for each experiment and generate:
   - Training and validation loss curves
   - Test accuracy results
   - Bias–Variance analysis plots showing the effect of model complexity on performance.



***
***
***


# Phase 3: Advanced Optimization

## Overview
In this phase, we explore advanced optimization algorithms beyond standard SGD. We implement and compare different gradient-based optimizers and investigate second-order optimization methods (Newton's method) to deepen our understanding of loss landscape dynamics.

This phase consists of the following tasks:
- **Task 3-1:** Implementing and comparing various optimizers: **SGD with Momentum, Nesterov, AdaGrad, RMSprop, and Adam**.
- **Task 3-2 (Conceptual):** Theoretical analysis of Newton's method, Hessian-vector products, and Conjugate Gradient.
- **Task 3-3 (Optional):** Implementing Newton's method with **Hessian-vector product (HVP)** approximation and comparing its convergence against SGD.

---

## Requirements
The following libraries are required to run the code:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

---

## Dataset Setup

This phase continues to use the **Fashion-MNIST** dataset. For consistency, we rely on the CSV format for easier processing with **Pandas** and **NumPy**.

### Required Files
Ensure the following files are located in your dataset directory:
- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

### Data Preparation
- **Normalization:** Pixel values are scaled to the range **[0, 1]**.
- **Vectorization:** Each 28×28 image is flattened into a **784-dimensional input vector**.
- **Loading:** The data loader is configured to handle the CSV structure efficiently.

---

## Implementation Details

### Architecture
- **Task 3-1 (Main Model):** We utilized a **3-layer MLP** based on the optimal configurations identified in Phase 2:
  - **Hidden Layers:** 3 layers with 256 neurons each
  - **Normalization:** Batch Normalization applied before the ReLU activation
  - **Initialization:** He Initialization for all weight matrices

- **Task 3-3 (TinyNet):** A simplified, low-parameter architecture (**1 hidden layer with 10 neurons**) was implemented to demonstrate the utility of **Newton-CG** using **Hessian-vector products**.

### Reproducibility & Robustness
All experiments for each optimizer are conducted over **3 independent runs** using different random seeds (**42, 43, 44**). This protocol ensures that the reported results (**Mean ± Standard Deviation**) are statistically meaningful and not dependent on a single random initialization.

---

## Key Observations

- **Optimizer Comparison (Task 3-1):** Adaptive methods like **Adam** and **RMSprop** demonstrated significantly faster initial convergence. The comparative results are derived from the **mean performance across 3 independent runs** to ensure reliability.

- **Convergence Speed:** Efficiency was measured by the number of epochs needed to reach an **85% validation accuracy threshold**. Results showed that while adaptive methods are faster, **Nesterov** and **SGD with Momentum** provide strong final stability.

- **Newton-CG vs SGD (Task 3-3):** While **Newton-CG** is theoretically attractive, the experiment highlighted its sensitivity in a deep learning context. Log-scale plots demonstrate that **SGD provides a more stable, monotonic convergence path**, whereas **Newton-CG** exhibited higher variance and sensitivity to numerical damping in the non-convex loss landscape.

---

## How to Run

1. Ensure the **Fashion-MNIST** dataset files are in the dataset directory.

2. Execute the **Phase 3 integrated script**:
   - The `run_phase3()` function orchestrates the learning rate search and the triple-run statistical evaluation for all **5 optimizers**.
   - The **Newton-CG** comparison experiment is handled via a dedicated script for `TinyNet`.

3. The execution will automatically generate:
   - **Validation Loss Curves:** Overlaid plots for all major optimizers
   - **Summary Table:** Reporting **Mean ± STD** for Test Accuracy and **Epochs to 85%**
   - **Newton-CG Comparison Plot:** A log-scale visualization comparing second-order and first-order convergence behavior



***
***
***


## Phase 4: Final Model & Generalization Analysis

### Overview
In the final phase, we integrate the best-performing techniques from previous stages—including **He Initialization, Batch Normalization, Adam Optimization, and L2 Regularization**—to build a robust deep MLP. The goal is to find an optimal architecture through random search and perform a rigorous analysis of the model's generalization capabilities.

This phase consists of:
- **Task 4-1:** Constructing a 5-layer deep network and optimizing its hyperparameters.
- **Task 4-2:** Performing a Bias-Variance decomposition and analyzing the "Generalization Gap."

---

### Dataset Setup
The final evaluation is conducted on the **Fashion-MNIST** dataset.
- **Data Split:** 50,000 samples for Training and 10,000 for Validation (Stratified).
- **Preprocessing:** Pixel values scaled to **[0, 1]**.
- **Efficiency:** DataLoaders configured with `batch_size=64` and `pin_memory=True`.

---

### Implementation Details

#### 🧠 Winner Architecture (Random Search Result)
After conducting a Random Search, the following 5-layer architecture was identified as the optimal configuration:
- **Input Layer:** 784 neurons (28x28 images).
- **Hidden Layer 1:** 512 neurons + BatchNorm + ReLU.
- **Hidden Layer 2:** 512 neurons + BatchNorm + ReLU.
- **Hidden Layer 3:** 512 neurons + BatchNorm + ReLU.
- **Hidden Layer 4:** 128 neurons + BatchNorm + ReLU.
- **Hidden Layer 5:** 512 neurons + BatchNorm + ReLU.
- **Output Layer:** 10 neurons (Softmax Cross-Entropy).

#### ⚙️ Optimized Hyperparameters
To achieve maximum stability and accuracy, the following parameters were applied:
- **Optimizer:** Adam
- **Learning Rate:** 0.00017
- **Weight Initialization:** He Initialization (Kaiming Normal)
- **Regularization:** Weight Decay ($1 \times 10^{-4}$)
- **Training Duration:** 20 Epochs
- **Batch Size:** 64

#### 📊 Statistical Significance
To ensure reliability, the winner architecture is trained **3 times** with different seeds. We report the **Mean Test Accuracy** and **Standard Deviation** to verify that the performance is consistent and not a result of "lucky" initialization.

---

### Key Observations
1. **Generalization Gap:** The loss curves monitor the gap between training and validation. A narrow gap confirms that the combination of **BatchNorm** and **Weight Decay** successfully prevents overfitting despite the model's depth.
2. **Bias-Variance Trade-off:** 
   - **Low Bias:** The 5-layer depth allows the model to capture high-level features of the Fashion-MNIST classes.
   - **Variance Control:** Managed via L2 regularization, ensuring high accuracy on the unseen Test set.
3. **Data Impact Analysis:** Reducing training samples to 10,000 would significantly increase **Variance**, leading to a wider generalization gap and requiring more aggressive regularization (like Dropout or higher Weight Decay).

---

### How to Run
> **Note:** Ensure the **"Reproducibility Setup"** cell is executed first to initialize the environment and device settings.

1. **Random Search:** Run the search loop to validate the architecture selection.
2. **Final Training:** Execute the 3-run statistical training block.
3. **Visualization:** The code will automatically output:
   - **Loss Evolution Plot:** Visualizing the Training vs. Validation loss.
   - **Generalization Gap:** Highlighted area indicating the model's stability.
   - **Final Performance:** Mean Accuracy and Standard Deviation across runs.
