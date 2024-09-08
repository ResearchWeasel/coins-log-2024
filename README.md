# COINs: Knowledge Graph Inference with Model-based Acceleration

LoG 2024 conference submission

---------

## Abstract

We present a new method, called COINs for **CO**mmunity **IN**formed graph embedding**s**, to enhance the efficiency of
knowledge graph models for link prediction and conjunctive query answering. COINs uses a community-detection-based graph
data augmentation and a two-step prediction pipeline: we first achieve node localization through community prediction,
and subsequently, we further localize within the predicted community. We establish theoretical criteria to evaluate our
method in our specific context and establish a direct expression of the reduction in time complexity. We empirically
demonstrate an important scalability-performance trade-off where for the median evaluation sample we preserve 97.18% of
the baseline accuracy in single-hop query answering, for only 7.52% of the original computational cost on a
single-CPU-GPU machine.

## Instructions

### Obtaining the code

Clone this repository by running:

`git clone https://github.com/ResearchWeasel/coins-log-2024.git`

### Dependencies

The implementation requires version `3.6.13` of the Python programming language.
To install it and the dependent Python packages, we recommend having [Anaconda](https://www.anaconda.com/download), then
running the following commands from the main directory of the repository files:

1. `conda create --name coins python=3.6.13`
2. `conda activate coins`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH='.'`

### Reproducing results

To regenerate the tables and figures provided in the paper, run the Jupyter notebook `graph_completion/Plotting.ipynb`.

Figure PDFs will be saved to the `graph_completion/results` directory.

To run end-to-end a COINs training and evaluation experiment from the paper, run the following command from the main
directory of the repository files:

- GPU:

  `CUDA_VISIBLE_DEVICES=<GPU ID> python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

- CPU (after setting the `device` config parameter to `cpu` in the YAML file):

  `python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

The experiment results will be saved to a directory in `graph_completion/results/<DATASET>/runs`.

----------

## Authors

- Anonymous
