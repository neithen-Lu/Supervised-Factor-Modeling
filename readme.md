# Readme

### Create virtual environment

Make sure you have conda installed and run the following command

```
conda create -n virtualenv python=3.9
conda activate virtualenv
pip install -r requirements.txt
```

To run R scripts, please install R package "bigtime" and "modelconf".

To run the Julia scripts, please install the following Julia package: Pkg,Distributions,TensorToolbox,LinearAlgebra,DelimitedFiles,SparseArrays,CSV,DataFrames

### Sidenotes

Note that our BVAR code comes from: http://joshuachan.org/code/code_large_BVAR.html and we choose Bayesian VAR with a natural shrinkage prior given by the function **BVAR-NCP**. Since the code is available online, we did not include it here.

(I) Note that we transform all sequences to be stationary and standardize them to zero mean and unit variance following McCracken and Ng (2020). Detailed transformations are provided in the supplementary file. Due to the transformation, we truncate the first three timepoints. The data after-transformation and truncation are provided in data/qd_large for the large size dataset, data/qd_medium for the medium size dataset and data/qd_small for the small size dataset.

### To reproduce the empirical results for the macroeconomic dataset:

Step 1: Run `batch_real.py` and set exp_name = 'qd_large'  for large size dataset or  'qd_medium' for medium size dataset or  'qd_small' for small size dataset

Note that when running "ours" model, the runtime warning from tensorly can be ignored.

Step 2: To see the results by running `vis.ipynb`

Step 3: Once the results are collected into a csv file similar to "result/large_forecast_err_l2.csv" run `mcs.R` to obtain the MCS p-values.

### To reproduce our simulation forecasting results:

Step 1: Run `batch_sim.py` and view the results in the "result" folder.

Step 2 (optional): If you would like to run results for VAR (MLR), VAR (SHORR) and BVAR, you would need to first generate the data using the following command line

- For VAR:

  ```sh
  python generate_forecast_data.py --dgp season_ar_diffrank --season 4 --dgp_p 4 --dgp_r1 4 --dgp_r2 2 --rho 0.9 --n_rep 100 --T 1200 --T0 34 --N 100 --exp_name var_forecast --seed 0
  ```
- For VARMA:

  ```sh
  python generate_forecast_data.py --dgp arma --dgp_p 1 --dgp_r 4 --rho 0.9 --n_rep 100 --T 2000 --T0 44 --N 100 --exp_name varma_forecast --seed 0
  ```

Step 2: Once the results are collected into a csv file similar to "result/large_forecast_err_l2.csv" run `mcs.R` to obtain the MCS p-values.

Note that the rate experiments can be run using  `batch_sim.py`  with model_list=["ours"] but the settings would need to be adjusted accordingly.
