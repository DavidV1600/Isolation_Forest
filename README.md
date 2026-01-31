# How to run

## 1) Install
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```
## 2) Put datasets

## 3) Run replicate_paper.py
Cd to the base directory and run:
```bash
python replicate_paper.py
```
This will run all experiments and generate results in the `outputs/` folder.
## 4) Generate plots and tables
To generate plots and tables for the paper, run:
```bash
python analyze_results.py
```
This will generate plots in `outputs/`.
## 5) Run sensitivity analysis
To run the parameter sensitivity analysis, run:
```bash
python sensitivity.py
```
The results will be saved in `outputs/parameter_sensitivity.csv`.
