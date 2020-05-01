__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr


def summary(y_pred, y_test):

    # show first 25 results
    print("\nSummary of results\n")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results.head(25))

    # evaluation
    print("\nEvaluation results\n")
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Spearman correlation:', spearmanr(y_test, y_pred))

    # NDCG score
    true_relevance = np.asarray([y_test])
    scores = np.asarray([y_pred])
    print('NDCG score: ', ndcg_score(true_relevance, scores))


