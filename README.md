# Kernel SVM

### As part of the code release assignment for DT558 Class.

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

The loss function is as below:

![Loss](images/formula.png)

The gradien is as below:

![Grad](images/grad2.png)

TODO:

- [x] Add `sklearn_compare.py`
- [ ] Fix `sim_demo.py` to evaluate SVM.
- [ ] Put all plots in 1 figure.
- [ ] Fix plots legend to include train_cache information.
- [x] Put training configurations in SVM init.
- [x] PEP8 `source/*`.
- [x] PEP8 `examples/*`.
- [x] README on how to use `src/svm`, run `examples/*`
