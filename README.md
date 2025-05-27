# Metagenome-Analysis
Practices of Metagenomics courses.

The two Python scripts stored in the `code` use `XGBoost` and `SVM` training models to predict and evaluate classification results. The notebook `test.ipynb` contains the original Random Forest model for prediction and evaluation.

When you need to introduce `CLR transformations`, just add a few lines after the feature alignment in the script:
```python
def clr_transform(df):
    pseudocount = 1e-6
    log_df = np.log(df + pseudocount)
    gm = log_df.mean(axis=1)
    clr = log_df.sub(gm, axis=0)
    return clr

X_train = clr_transform(X_train)
X_test = clr_transform(X_test)
```

The results show that CLR transformation has a positive effect on improving the prediction performance of the model within a certain range.
