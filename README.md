# ee046211-deep-learning

<h1 align="center">
  <br>
Technion EE 046211 - Deep Learning
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://github.com/royg27">Roy Ganz</a> •
    <a href="https://sites.google.com/danielsoudry">Daniel Soudry</a>
  </p>

Jupyter Notebook tutorials for the Technion's EE 046211 course "Deep Learning"
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046211-deep-learning"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046211-deep-learning/tree/main/"><img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046211-deep-learning/main"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>
<p align="center">
    <a href="https://taldatech.github.io/ee046211-deep-learning/">Student Projects (Spring 2021)</a>
  </p>



- [ee046211-deep-learning](#ee046211-deep-learning)
  * [Agenda](#agenda)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)


## Agenda

|File       | Topics Covered |
|----------------|---------|
|`Setting Up The Working Environment.pdf`| Guide for installing Anaconda locally with Python 3 and PyTorch, integration with PyCharm and using GPU on Google Colab |
|`ee046211_tutorial_01_machine_learning_recap.ipynb/pdf`| Supervised and Unsupervised Learning, Model Evaluation, Bias-Variance Tradeoff, Feature Scaling, Linear Regression, Gradient Descent, Regularization (Ridge, LASSO)|
|`ee046211_tutorial_02_single_neuron_recap.ipynb/pdf`| Discriminative models, Perceptron, Logistic Regression (also in PyTorch), Softmax Regression, Activation functions|
|`ee046211_tutorial_03_optimization_gradient_descent.ipynb/pdf`|Unimodal functions, Convexity, Hessain, Gradient Descent, SGD, Learning Rate, LR Scheculing / Annealing, Momentum, Nesterov Momentum, Adaptive Learning Rate Methods, Adagrad, RMSprop, Adam|
|`ee046211_tutorial_04_differentiation_autograd.ipynb/pdf`|Lagrange Multipliers, Automatic Differentiation (AutoDiff) Forward Mode and Reverese Mode, PyTorch Autograd|
|`ee046211_tutorial_05_multilayer_nn.ipynb/pdf`|Multi-Layer Perceptron (MLP), Backpropagation, Neural Netwroks in PyTorch, Weights Initialization - Xavier (Glorot), Kaiming (He), Deep Double Descent|
|`ee046211_tutorial_06_convnets_visual_tasks.ipynb/pdf`|2D Convolution (Cross-corelation), Convolution-based Classification, Convolutional Neural Networks (CNNs), Regularization and Overfitting, Dropout, Data Augmentation, CIFAR-10 dataset, Visualizing Filters, Applications of CNNs, The problems with CNNs (adversarial attacks, poor generalization, fairness-undesirable biases)|
|`ee046211_tutorial_07_sequential_tasks_rnn.ipynb/pdf`|Sequential Tasks, Natural Language Processing (NLP), Langiage Model, Perplexity, BLEU,  Recurrent Neural Network (RNN), Backpropagation Through Time (BPTT), Long Term Short Memory (LSTM), Gated Recurrent Unit (GRU), (Self Multi-head) Attention, Transformer, BERT and GPT, Teacher Forcing, torchtext, Sentiment Analysis|
|`ee046211_tutorial_08_training_methods.ipynb/pdf`|Feature Scaling, Normalization, Standardization, Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization, Vanishing Gradients, Exploding Gradients, Skip-Connection, Residual Nlock, ResNet, DenseNet, U-Net, Hyper-parameter Tuning: Grid Search, Random Search, Bayesian Tuning, Optuna with PyTorch|
|`ee046211_tutorial_09_self_supervised_representation_learning.ipynb/pdf`|Transfer Learning, Domain Adaptation, Pre-trained Networks, Sim2Real, BERT, Representation Learning, Self-Supervised Learning, Autoencoders, Contrastive Learning, Contrastive Predictive Coding (CPC), Simple Framework for Contrastive Learning of Visual Representations (SimCLR), Momentum Contrast (MoCo), Bootstrap Your Own Latent (BYOL)|
|`ee046211_tutorial_10_compression_pruning_amp.ipynb/pdf`|Resource Efficiency in DL, Automatic Mixed Precision (AMP), Quantization (Dynamic, Static), Quantization Aware Training (QAT), Pruning, The Lottery Ticket Hypothesis|


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

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
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
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
|`torchtext`| `conda install -c pytorch torchtext`|


5. To open the notebooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.
