# Set the default goal so running `make` with no arguments prints the help menu

.DEFAULT_GOAL := help
environment := ft_log_reg


.PHONY: help
help: ## Show this help menu
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: describe_test
describe_test: ## Show descriptive analysis of dataset_test.csv
	python3 Analysis/describe.py dataset_test.csv

.PHONY: describe_train
describe_train: ## Show descriptive analysis of dataset_train.csv
	python3 Analysis/describe.py dataset_train.csv

.PHONY: histogram_test
histogram_test: ## Show histogram of dataset_test.csv
	python3 Visualization/histogram.py dataset_test.csv

.PHONY: histogram_train
histogram_train: ## Show histogram of dataset_train.csv
	python3 Visualization/histogram.py dataset_train.csv

.PHONY: pair_plot_test
pair_plot_test: ## Show pair_plot of dataset_test.csv
	python3 Visualization/pair_plot.py dataset_test.csv

.PHONY: pair_plot_train
pair_plot_train: ## Show pair_plot of dataset_train.csv
	python3 Visualization/pair_plot.py dataset_train.csv

.PHONY: scatter_test
scatter_test: ## Show scatter of dataset_test.csv
	python3 Visualization/scatter.py dataset_test.csv

.PHONY: scatter_train
scatter_train: ## Show scatter of dataset_train.csv
	python3 Visualization/scatter.py dataset_train.csv


.PHONY: train
train: ## Train multi-classifier using a logistic regression one-vs-all approach
	python3 Regression/logreg_train.py dataset_train.csv
	
.PHONY: predict
predict: ## Predict houses for dataset_test.csv using weights.json
	python3 Regression/logreg_predict.py dataset_test.csv weights.json

.PHONY: set
set: ## Set a python environment for this project
	## bash ;	python3 -m venv $(environment); . ./$(environment)/bin/activate; pip install -r requirements.txt
	##bash -c "python3 -m venv $(environment) && source $(environment)/bin/activate && pip install -r requirements.txt"
	bash -c "python3 -m venv $(environment) && source $(environment)/bin/activate && pip install --upgrade pip &&pip install -r requirements.txt"

.PHONY: activate
activate: ## Activate the python environment for this project
	@echo "Run: source $(environment)/bin/activate"

.PHONY: unset
unset: ## removes the python 
	rm -rf $(environment)

.PHONY: upgrade
upgrade: ## Upgrades pip
	pip install --upgrade pip