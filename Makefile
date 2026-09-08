# Set the default goal so running `make` with no arguments prints the help menu

.DEFAULT_GOAL := help
environment := ft_log_reg
TRAIN_PCT ?= 80
EPOCHS ?= 3000
LEARNING_RATE ?= 0.01
TOLERANCE ?= 1e-6

.PHONY: help
help: ## Show this help menu
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: describe
describe: ## Show descriptive analysis of dataset_test.csv
	python3 Analysis/describe.py datasets/dataset_train.csv
	python3 Analysis/describe_pandas.py datasets/dataset_train.csv
	#diff -u dataset_test_describe_pandas.txt dataset_test_describe.txt


.PHONY: truants
truants: ## Studies NaN values
	python3 Analysis/truants.py datasets/dataset_train.csv

.PHONY: histogram
histogram: ## Show histogram of dataset_train.csv
	python3 Visualization/histogram.py datasets/dataset_train.csv

.PHONY: boxplot
boxplot: ## Show boxplot of dataset_train.csv
	python3 Visualization/boxPlot.py datasets/dataset_train.csv

.PHONY: pair_plot_test
pair_plot_test: ## Show pair_plot of dataset_test.csv
	python3 Visualization/pair_plot.py datasets/dataset_test.csv

.PHONY: pair_plot_train
pair_plot_train: ## Show pair_plot of dataset_train.csv
	python3 Visualization/pair_plot.py datasets/dataset_train.csv

.PHONY: scatter_test
scatter_test: ## Show scatter of dataset_test.csv
	python3 Visualization/scatter.py datasets/dataset_test.csv

.PHONY: scatter_train
scatter_train: ## Show scatter of dataset_train_normalized.csv
	python3 Visualization/scatter.py datasets/dataset_train_normalized.csv

.PHONY: split
split: ## Split dataset_train.csv into train and validation sets
	python3 Regression/split.py datasets/dataset_train_normalized.csv $(TRAIN_PCT)

.PHONY: train
train: ## Train multi-classifier using a logistic regression one-vs-all approach
	python3 Regression/train.py datasets/dataset_train_normalized_to_train.csv $(EPOCHS) $(LEARNING_RATE) $(TOLERANCE)

.PHONY: test
test: ## Test multi-classifier using a logistic regression one-vs-all approach
	python3 Regression/test.py datasets/dataset_train_normalized_to_test.csv datasets/weights.json

.PHONY: predict
predict: ## Predict houses for dataset_test.csv using weights.json
	python3 Regression/predict.py datasets/dataset_test.csv datasets/weights.json datasets/dataset_train_describe.csv

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

.PHONY: norminette
norminette: ## Run norminette on all .py files
	flake8  */*.py		
