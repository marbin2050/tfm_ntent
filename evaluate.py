__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# def rfecv():
#


def summary(y_pred, y_test, process_title):

    # evaluation
    print("\nEvaluation results: " + process_title + "\n")
    print("R2 score ", r2_score(y_test, y_pred))
    print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Spearman correlation:', spearmanr(y_test, y_pred))
    print('Pearson correlation:', pearsonr(y_test.flatten(), y_pred))
