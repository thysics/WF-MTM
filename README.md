# Wright-Fisher Multimorbidity Trajectory Model

This code implements the Wright-Fisher Multimorbidity Trajectory Model (WF-MTM) to dataset.
The repository consists of 1. MCMC sampler 2. Implementation template and 3. Store folders for the MCMC outcome.

## Project set-up

This repository requires the following python packages:
1. numpy
2. mathplot.pylab

## How to run locally

To understand how one can implement this algorithm, see `UnderModel/UM01/UM01_MCMC.ipynb`.
The above juypter code applies the WF-MTM to the simulated data by
1. Import MCMC sampler from Sampler/
2. Initialise latent variables / parameters by running `UM01_Init.ipynb`
3. Run MCMC algorithm for several thousand iterations while saving the posterior samples in MCMC/UM01/HB_R2/
4. Analyze the posterior inference outcome using `UnderModel/UM01/UM01_ANALYSIS.ipynb`

## Reference

[Kim, Woojung, Paul A. Jenkins, and Christopher Yau. "Feature Allocation Approach for Multimorbidity Trajectory Modelling." Machine Learning for Health. PMLR, 2022.](https://proceedings.mlr.press/v193/kim22a/kim22a.pdf)
