### Workflow

1) Retrieve latest data from crunchbase in notebook `fetch_crunchbase_data.ipynb`
2) Train model on data and predict missing values in notebook `pmv_regression.ipynb`
3) Compute returns in `compute_returns.ipynb`
4) Run `main.ipynb` to run likelyhood optimization


### TODO
1) Enhance data further with more exit values, and money raised during exit (between steps 2 and 3)
2) Enhance prediciton model with additional features, perform context-dependent feature selection
3) Enhance Cochrane model
4) More stylized facts, more market analysis