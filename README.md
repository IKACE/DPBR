# DPBR: Combining Data Privacy and Byzantine Resilience in Distributed Machine Learning

In this research project, we are focusing on simultaneously tolerating Byzantine workers and preserving data privacy in distributed machine learning. This repository stores source code for running the experiments and produced results.

We start by studying and developing theoretical upper bounds for combining data privacy and Byzantine resilience. We use Gaussian noise injection to provide privacy for the gradients submitted to the master. Experiments based on theoretical findings show that batch size is the key factor to ensure training convergence of the SGD algorithm. We then conduct further experiments to study the effects of using alternative data privacy measures such as gradient sparsification. At last, we revisit the batch size from a theoretical view and demonstrate that the reason why batch size is important lies in its effect on data privacy instead of guaranteeing Byzantine resilience. In light of this observation, we argue that while a large batch size is important for the model to converge in the context of combining Data Privacy and Byzantine resilience, the demand on the batch size is not unlimited; instead, a certain (even a large constant) batch size is good enough: it does not need to unboundedly grow with model's parameter size.

# Environment Setup
Creating a new python environment and call `pip install -r requirements.txt` will set up the envrionment.
In case of errors regarding Gtk, install `PyGObject` following the instructions at https://pygobject.readthedocs.io/en/latest/getting_started.html.

# Reproducing Results
The experiments are divided into 2 parts: combining data privacy and Byzantine resilience for convolutional neural network (CNN) and reconstructing original data from gradients.

## Data Privacy and Byzantine Resilience for CNNs

Run ./DifferentialByzantine/reproduce.py to reproduce accuracy and loss results for CNNs. Set privacy-epsilon argument to be 0.5 to enable gradient sparsification, set it between (0, 1) to enable Gaussian noise injection for differential privacy, and set it to None to disable differential privacy. 

## Reconstructing original data from gradients

Run ./dlg/single_batch_main.py to reconstruct a single batch image from CIFAR-100.

Run ./dlg/main.py to reconstruct a selected image from a batch of customized size. 

Run ./invertinggradients/ResNet32-10\ -\ Recovering\ 100\ CIFAR-100\ images.ipynb to reconstruct a batch of 100 images from CIFAR-100, one for each class.

# Result Plots
## Byzantine Resilience of CNN without Data Privacy Measures
![](/plots/591_cnn_no_noise.png)

## Byzantine Resilience of CNN with Gaussian Noise Injection
![](/plots/591_cnn_gauss_noise.png)

## Byzantine Resilience of CNN with Gradient Sparsification
![](/plots/591_cnn_grad_sparse.png)

## Image Reconstruction
![](/plots/591_image_reconstruct.png)
