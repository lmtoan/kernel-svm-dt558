# Kernel SVM

## Usage

For simulated data demo, try:
```
cd examples/
python sim_demo.py
```

For DIGITS data demo, try:
```
cd examples/
python real_demo.py
```
## Dependencies 
Python 3 or 2.7 with the following packages installed:
1. Numpy
2. Scipy
3. Scikit-learn
4. Pandas
5. Matplotlib
6. tqdm

## Background

The loss function is as below:

![Loss](images/formula.png)

The gradient is as below:

![Grad](images/grad2.png)

## Plots

![r](images/rbf.png)

![l](images/linear.png)

![p](images/poly_7.png)

![real](images/multiclass_demo.png)

## Screenshots

![1](images/sim_demo.png)

![2](images/real_demo.png)

![3](images/sklearn_compare.png)

