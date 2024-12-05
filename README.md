
# LeNet-MiniPlaces

LeNet-MiniPlaces is a project that implements the LeNet-5 Convolutional Neural Network (CNN) architecture and customizes it for the MiniPlaces scene recognition dataset. This project explores deep learning techniques using PyTorch and involves training and evaluating models under different configurations.

---

## Project Overview

- **Goal**: 
  - Implement and train the LeNet-5 CNN architecture.
  - Understand trainable parameters and explore how hyperparameters affect training outcomes.
  - Design a custom CNN for scene recognition using the MiniPlaces dataset.

- **Dataset**: MiniPlaces, a subset of the Places2 dataset, containing:
  - **100,000** training images
  - **10,000** validation images
  - **10,000** testing images
  - Images are resized to 32x32 for training efficiency.

- **Technologies**: 
  - Python
  - PyTorch
  - MiniPlaces dataset
  - Conda for environment management

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/LeNet-MiniPlaces.git
cd LeNet-MiniPlaces
```

### 2. Set Up the Environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create and activate the project environment:
   ```bash
   conda create -n lenet-miniplaces pytorch torchvision torchaudio tqdm cpuonly -c pytorch
   conda activate lenet-miniplaces
   ```

### 3. Dataset Preparation

1. Download the dataset (MiniPlaces) from [here](http://places2.csail.mit.edu/) or use the provided backup link.
2. Unpack the dataset:
   ```bash
   tar -xvf data.tar.gz
   mkdir -p data/miniplaces && mv images/* data/miniplaces/
   ```
3. Update the `MiniPlaces` constructor to `download=False` in all scripts.

---

## Training and Evaluation

### 1. LeNet-5 Implementation

The LeNet-5 architecture is implemented as follows:

- **Conv Layer 1**: 6 output channels, kernel size 5, stride 1, ReLU activation, MaxPooling (2x2).
- **Conv Layer 2**: 16 output channels, kernel size 5, stride 1, ReLU activation, MaxPooling (2x2).
- **Fully Connected Layers**:
  - FC1: Output dimension 256, ReLU activation.
  - FC2: Output dimension 128, ReLU activation.
  - FC3: Output dimension = 100 (number of classes).

### 2. Hyperparameter Configurations

Train the model under the following configurations:
- Default settings
- Batch sizes: 8, 16
- Learning rates: 0.05, 0.01
- Epochs: 20, 5

Scripts:
- Training: `train_miniplaces.py`
- Evaluation: `eval_miniplaces.py`

### 3. Running the Training

Train the model:
```bash
python train_miniplaces.py
```

Evaluate the model:
```bash
python eval_miniplaces.py --load ./outputs/model_best.pth.tar
```

Results are saved in `results.txt`.

---

## Key Features

- Implementation of a foundational CNN architecture (LeNet-5).
- Support for training with adjustable hyperparameters.
- Accurate validation metrics.
- Customizable network design for scene recognition.

---

## File Structure

```
LeNet-MiniPlaces/
│
├── data/                   # Dataset directory
├── outputs/                # Model checkpoints and evaluation outputs
├── train_miniplaces.py     # Training script
├── eval_miniplaces.py      # Evaluation script
├── dataloader.py           # Data loading utilities
├── student_code.py         # Core LeNet-5 implementation
└── README.md               # Project documentation
```

---

## References

- **LeNet-5**: [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)
- **MiniPlaces Dataset**: [MiniPlaces Dataset](http://places2.csail.mit.edu/)
- **PyTorch**: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

If you encounter issues or have questions, feel free to open an issue on this repository!

**Happy Learning!**
