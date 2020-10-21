
# Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks

This repository is the official implementation of Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks

This work is received by NeurIPS 2020.

## Requirements

To run the code, please download the pytorch >= 1.0 with torchvision


## Training

To train the model(s) in the paper, run this command:

```train
python main.py --method fireflyn --model vgg19
```

You can also try different growing method [exact/fast/fireflyn/random] which represent original splitting, fast splitting, firefly splitting, NASH described in the paper.

## TODO
Detail tutorial and more experiments in the paper are under construction, will release in the future.
