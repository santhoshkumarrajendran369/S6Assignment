# ERA V3 Assignment 6 - MNIST Classification

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for MNIST digit classification.

## Model Architecture

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
         MaxPool2d-3           [-1, 16, 14, 14]               0
           Dropout-4           [-1, 16, 14, 14]               0
            Conv2d-5           [-1, 16, 14, 14]           2,320
       BatchNorm2d-6           [-1, 16, 14, 14]              32
            Conv2d-7           [-1, 16, 14, 14]           2,320
       BatchNorm2d-8           [-1, 16, 14, 14]              32
         MaxPool2d-9             [-1, 16, 7, 7]               0
          Dropout-10             [-1, 16, 7, 7]               0
           Conv2d-11             [-1, 32, 7, 7]           4,640
      BatchNorm2d-12             [-1, 32, 7, 7]              64
           Conv2d-13             [-1, 32, 7, 7]           9,248
      BatchNorm2d-14             [-1, 32, 7, 7]              64
           Conv2d-15             [-1, 10, 7, 7]             330
================================================================
Total params: 19,242
Trainable params: 19,242
Non-trainable params: 0
----------------------------------------------------------------
```

## Training Configuration

- **Optimizer**: SGD with momentum
  - Learning Rate: 0.05
  - Momentum: 0.9
  - Weight Decay: 5e-4
- **Learning Rate Scheduler**: Cosine Annealing
- **Batch Size**: 128
- **Epochs**: 15
- **Data Augmentation**:
  - Random Rotation (-7° to 7°)
  - Random Affine Translation (±10%)
  - Normalization (mean=0.1307, std=0.3081)

## Test Logs

| Epoch | Test Loss | Test Accuracy |
|-------|-----------|---------------|
| 1     | 0.0606    | 98.17%        |
| 2     | 0.0694    | 98.00%        |
| 3     | 0.0456    | 98.59%        |
| 4     | 0.0539    | 98.29%        |
| 5     | 0.0387    | 98.84%        |
| 6     | 0.0328    | 98.91%        |
| 7     | 0.0260    | 99.13%        |
| 8     | 0.0258    | 99.15%        |
| 9     | 0.0312    | 98.99%        |
| 10    | 0.0293    | 99.07%        |
| 11    | 0.0192    | 99.41%        |
| 12    | 0.0189    | 99.44%        |
| 13    | 0.0173    | 99.48%        |
| 14    | 0.0178    | 99.46%        |
| 15    | 0.0165    | 99.46%        |

**Best Test Accuracy: 99.48%** (Epoch 13)

## Training Progress

The model showed consistent improvement throughout training:
1. Achieved >98% accuracy in the first epoch
2. Crossed 99% accuracy by epoch 7
3. Reached peak performance of 99.48% in epoch 13
4. Maintained stable performance in final epochs

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
torchsummary>=1.5.1
```

## Files

- `model.py`: Contains the CNN architecture
- `train.py`: Training script with data loading and training loop
- `assignment6.ipynb`: Jupyter notebook combining all code
- `requirements.txt`: Project dependencies

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training:
```bash
python train.py
```

Or open and run `assignment6.ipynb` in Jupyter Notebook.
