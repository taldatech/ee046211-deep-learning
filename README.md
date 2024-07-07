# ee046211-deep-learning

<h1 align="center">
  <br>
Technion ECE 046211 - Deep Learning
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/dl_intro_anim.gif" height="200">
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://sites.google.com/danielsoudry">Daniel Soudry</a>
  </p>

Jupyter Notebook tutorials for the Technion's ECE 046211 course "Deep Learning"

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046211-deep-learning"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046211-deep-learning/tree/main/"><img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046211-deep-learning/main"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>
<p align="center">
    <a href="https://taldatech.github.io/ee046211-deep-learning/">Student Projects Website</a> • <a href="https://www.youtube.com/playlist?list=PLy3Xsl9jz-9WBHO850WFxv2TB5qtAlk0r">Video Tutorials (Winter 2024)</a>
  </p>



- [ee046211-deep-learning](#ee046211-deep-learning)
  * [Agenda](#agenda)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)


## Agenda

|File       | Topics Covered | Video |
|----------------|---------|-------|
|`Setting Up The Working Environment.pdf`| Guide for installing Anaconda locally with Python 3 and PyTorch, integration with PyCharm and using GPU on Google Colab |-|
|`ee046211_tutorial_01_`<br>`machine_learning_recap.ipynb/pdf`| Supervised and Unsupervised Learning, Model Evaluation, Bias-Variance Tradeoff, Feature Scaling, Linear Regression, Gradient Descent, Regularization (Ridge, LASSO)|<a href="https://youtu.be/koB8k5gIzj0?si=8f6AUeyFLr4iG3em">Video Link</a>|
|`ee046211_tutorial_02_`<br>`single_neuron_recap.ipynb/pdf`| Discriminative models, Perceptron, Logistic Regression (also in PyTorch), Softmax Regression, Activation functions|<a href="https://youtu.be/qx9mgw_I628?si=edL9Fhs_E8K94f-r">Video Link</a>|
|`ee046211_tutorial_03_`<br>`optimization_gradient_descent.ipynb/pdf`|Unimodal functions, Convexity, Hessain, Gradient Descent, SGD, Learning Rate, LR Scheculing / Annealing, Momentum, Nesterov Momentum, Adaptive Learning Rate Methods, Adagrad, RMSprop, Adam, AdaBelief, MADGRAD, Adan, Schedule-free Optimization (SGD, Adam)|<a href="https://youtu.be/NguufamjuH4?si=I3v6Zi6VJKblt9d3">Video Link - Part 1</a><br><br><a href="https://youtu.be/3Yl2XMInUkE?si=JGtYN-6v9aNAbIIR">Video Link - Part 2</a>|
|`ee046211_tutorial_04_`<br>`differentiation_autograd.ipynb/pdf`|Lagrange Multipliers, Automatic Differentiation (AutoDiff) Forward Mode and Reverese Mode, PyTorch Autograd|<a href="https://youtu.be/hj-xKoYnQ7Y?si=KRiakLK8bhkm4HBb">Video Link</a>|
|`ee046211_tutorial_05_`<br>`multilayer_nn.ipynb/pdf`|Multi-Layer Perceptron (MLP), Backpropagation, Neural Netwroks in PyTorch, Weights Initialization - Xavier (Glorot), Kaiming (He), Deep Double Descent|<a href="https://youtu.be/lNBsPaVe1ss?si=YMMumgchkfCDymx-">Video Link</a>|
|`ee046211_tutorial_06_`<br>`convnets_visual_tasks.ipynb/pdf`|2D Convolution (Cross-correlation), Convolution-based Classification, Convolutional Neural Networks (CNNs), Regularization and Overfitting, Dropout, Data Augmentation, CIFAR-10 dataset, Visualizing Filters, Applications of CNNs, The problems with CNNs (adversarial attacks, poor generalization, fairness-undesirable biases)|<a href="https://youtu.be/5WEGszTK7yQ?si=bb7RvIABlODhpZBY">Video Link - Part 1</a><br><br><a href="https://youtu.be/s4C2r6YvKA0?si=lDzoSeEi_pVrwATu">Video Link - Part 2</a>|
|`ee046211_tutorial_07_`<br>`sequential_tasks_rnn.ipynb/pdf`|Sequential Tasks, Natural Language Processing (NLP), Language Model, Perplexity, BLEU,  Recurrent Neural Network (RNN), Backpropagation Through Time (BPTT), Long Term Short Memory (LSTM), Gated Recurrent Unit (GRU), RWKV, xLSTM, Multi-head Self-Attention, Transformer, BERT and GPT, Teacher Forcing, torchtext, Sentiment Analysis, Transformers Warmup, Intialization, GLU variants, Pre-norm and Post-norm, RMSNorm, SandwichNorm, ReZero, Rectified Adam (RAdam), Relative Positional Encoding/Embedding|<a href="https://youtu.be/Zm3mCSnaG_8?si=XFuE93DgzrPn0I4D">Video Link - Part 1</a><br><br><a href="https://youtu.be/raKxOKm76Mk?si=U_d-yRe0DifUfTqK">Video Link - Part 2</a><br><br><a href="https://youtu.be/sLkOjlEMsh0?si=vy5S5lcYX_QR7d7e">Video Link - Part 3</a>|
|`ee046211_tutorial_08_`<br>`training_methods.ipynb/pdf`|Feature Scaling, Normalization, Standardization, Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization, Vanishing Gradients, Exploding Gradients, Skip-Connection, Residual Nlock, ResNet, DenseNet, U-Net, Hyper-parameter Tuning: Grid Search, Random Search, Bayesian Tuning, Optuna with PyTorch|<a href="https://youtu.be/ZvBW758Ze0E?si=14yR4af1NSngXjVN">Video Link</a><br><br><a href="https://youtu.be/5f4sN0brzXE?si=tQTXZ6PC-KHB59OX">Video Link - Optuna Tutorial</a>|
|`ee046211_tutorial_09_`<br>`self_supervised_representation_learning.ipynb/pdf`|Transfer Learning, Domain Adaptation, Pre-trained Networks, Sim2Real, BERT, Low-rank Adaptation - LoRA, DoRA, Representation Learning, Self-Supervised Learning, Autoencoders, Contrastive Learning, Contrastive Predictive Coding (CPC), Simple Framework for Contrastive Learning of Visual Representations (SimCLR), Momentum Contrast (MoCo), Bootstrap Your Own Latent (BYOL), DINO, CLIP| <a href="https://youtu.be/DByiUrXFquU?si=gd3NFr256X1y6wXU">Video Link - Part 1 - Transfer Learning</a><br><br><a href="https://youtu.be/q115i8E6cr0?si=M6efB5MgvgsP4vNU">Video Link - Part 2 - Self-supervised Learning</a>|
|`ee046211_tutorial_10_`<br>`compression_pruning_amp.ipynb/pdf`|Resource Efficiency in DL, Automatic Mixed Precision (AMP), Quantization (Dynamic, Static), Quantization Aware Training (QAT), LLM Quantization, Pruning, The Lottery Ticket Hypothesis|<a href="https://youtu.be/BFZyr0MBAdA?si=UotQQ7_LR_6hD9zh">Video Link</a>|
|`pytorch_maximize_cpu_gpu_utilization.ipynb/pdf`|Tips and Tricks for efficient coding in PyTorch, Maximizing the CPU and GPU utilization, `nvidia-smi`, PyTorch Profiler, AMP, Multi-GPU training, HF Accelerate, RL libraries|<a href="https://youtu.be/tIoa8axf9MI?si=-pMUFrk9NE2qnlKu">Video Link</a>|


## Running The Notebooks
You can view the tutorials online or download and run locally.

### Running Online

|Service      | Usage |
|-------------|---------|
|Jupyter Nbviewer| Render and view the notebooks (can not edit) |
|Binder| Render, view and edit the notebooks (limited time) |
|Google Colab| Render, view, edit and save the notebooks to Google Drive (limited time) |


Jupyter Nbviewer:

[![nbviewer](https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/)


Press on the "Open in Colab" button below to use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis)

Or press on the "launch binder" button below to launch in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master)

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046211-deep-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.


## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/download
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `deep_learn`. If you did this, you will only need to install PyTorch, see the table below.
3. Alternatively, you can create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`optuna`| `pip install optuna`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|
|`torchtext`| `conda install -c pytorch torchtext`|
|`torchdata`| `conda install -c pytorch torchdata` + `pip install portalocker`|


5. To open the notebooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.
