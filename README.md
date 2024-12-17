# Spatial Frequency Maps in Human Visual Cortex: </br>A Replication and Extension

This repository includes codes to reproduce the analysis and figures to map spatial frequency preferences in human visual cortex using Natural Scenes Dataset - synthetic data.

for [spatial frequency preferences in human visual cortex](https://jov.arvojournals.org/article.aspx?articleid=2792643).

Citation
```
Ha, Broderick, Kay, & Winawer (2022). Spatial Frequency Maps in Human Visual Cortex: A Replication and Extension. .., .., . https://??

```

Table of Contents

* [Dependencies](#dependencies)
     * [Conda environment](#conda-environment)
* [Data](#data)
   * [Processed data](#processed-data)
   * [NSD synthetic data](#nsd-synthetic-data)
* [Analysis pipeline](#analysis-pipeline)
    * [Reproducing the figures](#reproducing-the-figures)
    * [Understanding the pipeline](#understanding-the-pipeline)

# Dependencies
All of the code in this repository is written in Python (3.7 and 3.8). To reproduce the python environment, we recommend using Conda to manage the dependencies.

## Conda environment 
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your
   system with the appropriate python version.
2. Install [mamba](https://github.com/mamba-org/mamba): `conda install mamba
  -n base -c conda-forge`
3. Download this github repository: `git clone git@github.com:JiyeongHa/Spatial-Frequency-Preference_NSDsyn.git`
4. move to where the repository is downloded: `cd path/to/sfp`
5. run `mamba env create -f environment.yml`.
6. Type `conda activate sfp` to activate this environment
   and all required packages will be available.
   
   
# Data 
## NSD synthetic data
The access to the NSD synthetic data will be granted after filling out the form on the NSD website (https://naturalscenesdataset.org/).

## processed data
The data for this project is available on OSF (https://osf.io/umqkw/).  

# Analysis pipeline
## Reproducing the figures
We used snakemake to manage the analysis pipeline. The pipeline is defined in the `Snakefile` in the root directory. To reproduce all the figures, you can use the following command:
Add `-N` if you wish to run the pipeline in dry-run mode.
```
snakemake -j1 figure_all

```

## Understanding the pipeline 
We also provide a set of jupyter notebooks to understand the pipeline. Under the `notebooks` directory, you can find the `pipeline.ipynb` notebook. This notebook provides a step-by-step guide to the analysis pipeline. The number indicates the order of the analysis steps.

