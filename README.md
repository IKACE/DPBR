# DPBR: Combining Data Privacy and Byzantine Resilience in Distributed Machine Learning

In this research project, we are focusing on simultaneously tolerating Byzantine workers and preserving data privacy in distributed machine learning.

# Environment Setup
Creating a new python environment and call `pip install -r requirements.txt` will set up the envrionment.
In case of errors regarding Gtk, install `PyGObject` following the instructions at https://pygobject.readthedocs.io/en/latest/getting_started.html.

# Reproducing Results
The experiments are divided into 2 parts: combining data privacy and Byzantine resilience for convolutional neural network (CNN) and reconstructing original data from gradients.

## Data Privacy and Byzantine Resilience for CNNs

Run ./DifferentialByzantine/reproduce.py to reproduce accuracy and loss results for CNNS. Set privacy-epsilon argument to be 0.5 to enable gradient sparsification, set it between (0, 1) to enable Gaussian noise injection for differential privacy, and set it to None to disable differential privacy. 

## Reconstructing original data from gradients

Run ./dlg/single_batch_main.py to reconstruct a single batch image from CIFAR-100.

Run ./invertinggradients/ResNet32-10\ -\ Recovering\ 100\ CIFAR-100\ images.ipynb to reconstruct a batch of 100 images from CIFAR-100, one for each class.