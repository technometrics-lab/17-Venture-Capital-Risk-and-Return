# 17-Venture-Capital-Risk-and-Return

This thesis revisits a maximum likelihood methodology first proposed by Cochrane (2005), to estimate the mean, standard deviation, alpha, and beta of venture capital investments, while correcting for the selec- tion bias. I use a new up-to-date dataset provided by Crunchbase, and enhance it to account for missing firm valuations using machine learn- ing. The model is able to infer, with a median error of less than 4%, the true log-value of the firm, for a total of nearly 120,000 observations, or six times more than the original paper, from 2010 to 2022. I find an annualized expected return of around 38%, robust to the choice of the benchmark and an annualized alpha of 32.14%, with a beta of 1.37, and very high idiosyncratic risk around 40%. Depending on the sector, I find a beta lower than 1 for the health industry and of up to 1.86 for the tech sector. The health industry exhibits the lowest alpha (24%) and the tech the highest (36%). I finally study the particular sub-sector of cyber- security firms and find an alpha of 36%, on par with the tech sector, but with a lower beta of 1.56.

# Contents

* data: contains all data necessary to run the main program, as well as test datasets from Cochrane's paper. Crunchbase raw data is not available by default but can be retrieved using the notebook `fetch_crunchbase_data.ipynb`
* profiling: contains HTML reports for the simulation performance benchmark, can be ignored
* src: contains all code and notebooks as well as result files

# How to

0. To start from scratch, run the code in `fetch_crunchbase_data.ipynb` to get up-to-date data from Crunchbsae. An API key is required in a `.env` file in the root folder
1. Run `notebooks/stylzed_facts.ipynb` to get a clean copy of the dataset and a complete analysis of the data
2. Run `notebooks/pmv_regression.ipynb` to train the models (are use the pre-trained ones) to complete the dataset
3. Run `notebooks/exit_value.ipynb` to grab all available exit values for the dataset
4. Run `notebooks/compute_returns.ipynb` to compute returns for the dataset
5. Run `scripts/main.py` with the correct arguments to perform the MLE. Values for T-bills and market returns are automatically fetched from APIs depending on detected start and end dates in the provided dataset and the specified ticker (such as ^SP500TR)

Figures are saved in `notebooks/figures` and models for PMV are stored in notebooks/models

Results are saved in `src\results` as pkl files. They contain a dictionnary with: start/end dates of the simulation, results as a pandas DataFrame with values and standard errors as rows, for all parameters. The method `print_results` from `scripts.utils.py` returns and prints inferred values for alpha, beta, E[R] and E[ln R].
