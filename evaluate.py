__author__ = '{Alfonso Aguado Bustillo}'

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score, recall_score, precision_score, \
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt


def roc_curve(y_test, y_pred):
    # ROC curve
    fpr_list, tpr_list, thresholds = roc_curve(y_test, y_pred, pos_label=2)
    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(fpr_list, tpr_list, lw=3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.show()


def confusion_matrix(y_test, y_pred, process_title):
    # Confusion matrix
    # get confusion matrix from sklearn
    cm = confusion_matrix(y_test, y_pred)
    # plot using matplotlib and seaborn
    ax = plt.figure(figsize=(10, 10))
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,
                                 as_cmap=True)
    sns.set(font_scale=2.5)
    sns.heatmap(cm, annot=True, cmap=cmap, cbar=False,
                xticklabels=['Unpopular', 'Popular'], yticklabels=['Unpopular', 'Popular'])
    plt.ylabel('Actual Labels', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.title(process_title)
    plt.show()


def summary(y_pred, y_test, regression=True, process_title="regression"):
    print("\nEvaluation results: " + process_title + "\n")

    # evaluation
    if regression:
        print("R2 score ", r2_score(y_test, y_pred))
        print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Spearman correlation:', spearmanr(y_test, y_pred))
        print('Pearson correlation:', pearsonr(y_test.flatten(), y_pred.flatten()))
    else:
        clf_results = {}

        acc_macro = round(accuracy_score(y_test, y_pred), 2)
        clf_results['acc_macro'] = acc_macro
        print("Accuracy (macro): ", acc_macro)
        prec_macro = round(precision_score(y_test, y_pred,  pos_label=1, average='macro'), 2)
        clf_results['prec_macro'] = prec_macro
        print("Precision score (macro): ", prec_macro)
        rec_macro = round(recall_score(y_test, y_pred,  pos_label=1, average='macro'), 2)
        clf_results['rec_macro'] = rec_macro
        print("Recall score: (macro)", rec_macro)
        f1_macro = round(f1_score(y_test, y_pred, average='macro'), 2)
        clf_results['f1_macro'] = f1_macro
        print("F1 score: ", f1_macro)
        auc_macro = round(roc_auc_score(y_test, y_pred, average="macro"), 2)
        clf_results['auc_macro'] = auc_macro
        print("AUC (macro): ", auc_macro)

        popular_class_prec, popular_class_rec, popular_class_f1, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1,
                                                                                         average='binary')
        clf_results['popular_class_prec'] = popular_class_prec
        clf_results['popular_class_rec'] = popular_class_rec
        clf_results['popular_class_f1'] = popular_class_f1
        print("Precision, Recall, F1 scores (popular class)", precision_recall_fscore_support(y_test, y_pred,
                                                                                              pos_label=1,
                                                                                              average='binary'))
        unpopular_class_prec, unpopular_class_rec, unpopular_class_f1, support =\
            precision_recall_fscore_support(y_test, y_pred, pos_label=0, average='binary')

        clf_results['unpopular_class_prec'] = unpopular_class_prec
        clf_results['unpopular_class_rec'] = unpopular_class_rec
        clf_results['unpopular_class_f1'] = unpopular_class_f1
        print("Precision, Recall, F1 scores (unpopular class)",
              precision_recall_fscore_support(y_test, y_pred, pos_label=0, average='binary'))

        weighted_prec, weighted_rec, weighted_f1, weighted_support = precision_recall_fscore_support(y_test, y_pred, pos_label=1,
                                                                                        average='weighted')
        clf_results['weighted_prec'] = weighted_prec
        clf_results['weighted_rec'] = weighted_rec
        clf_results['weighted_f1'] = weighted_f1
        auc_weighted = round(roc_auc_score(y_test, y_pred, average="weighted"), 2)
        clf_results['auc_weighted'] = auc_weighted
        print("AUC (weighted): ", auc_weighted)
        print("Average scores (weighted)", precision_recall_fscore_support(y_test, y_pred, pos_label=1,
                                                                           average='weighted'))

        # confusion_matrix(y_test, y_pred, process_title)
        # roc_curve(y_test, y_pred)

        return clf_results


def plots_summary(results, output_file_name='data/output_files/lgr_results.png'):

    metrics_names = results.keys()

    # macro emtrics
    f1_macro = []
    prec_macro = []
    rec_macro = []
    auc_macro = []

    # popular class metrics
    popular_class_f1 = []
    popular_class_prec = []
    popular_class_rec = []

    # unpopular class metrics
    unpopular_class_f1 = []
    unpopular_class_prec = []
    unpopular_class_rec = []

    # weighted metrics
    weighted_f1 = []
    weighted_pred = []
    weighted_rec = []
    auc_weighted = []

    for lgr_key in metrics_names:
        f1_macro.append(results.get(lgr_key)['f1_macro'])
        prec_macro.append(results.get(lgr_key)['prec_macro'])
        rec_macro.append(results.get(lgr_key)['rec_macro'])

        popular_class_f1.append(results.get(lgr_key)['popular_class_f1'])
        popular_class_prec.append(results.get(lgr_key)['popular_class_prec'])
        popular_class_rec.append(results.get(lgr_key)['popular_class_rec'])

        unpopular_class_f1.append(results.get(lgr_key)['unpopular_class_f1'])
        unpopular_class_prec.append(results.get(lgr_key)['unpopular_class_prec'])
        unpopular_class_rec.append(results.get(lgr_key)['unpopular_class_rec'])

        weighted_f1.append(results.get(lgr_key)['weighted_f1'])
        weighted_pred.append(results.get(lgr_key)['weighted_prec'])
        weighted_rec.append(results.get(lgr_key)['weighted_rec'])

        auc_macro.append(results.get(lgr_key)['auc_macro'])
        auc_weighted.append(results.get(lgr_key)['auc_weighted'])

    df = pd.DataFrame(list(zip(metrics_names, popular_class_f1)), columns=['bucket', 'popular_class_f1'])
    df['popular_class_prec'] = popular_class_prec
    df['popular_class_rec'] = popular_class_rec

    df['unpopular_class_f1'] = unpopular_class_f1
    df['unpopular_class_prec'] = unpopular_class_prec
    df['unpopular_class_rec'] = unpopular_class_rec

    df['weighted_f1'] = weighted_f1
    df['weighted_prec'] = weighted_pred
    df['weighted_rec'] = weighted_rec

    df['f1_macro'] = f1_macro
    df['prec_macro'] = prec_macro
    df['rec_macro'] = rec_macro

    df['auc_macro'] = auc_macro
    df['auc_weighted'] = auc_weighted

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 'x-large'
    plt.rcParams['ytick.labelsize'] = 'x-large'

    fig, axs = plt.subplots(5, 3)
    fig.tight_layout(pad=1)
    axs[0, 0].plot(df["bucket"], df["popular_class_f1"])
    axs[0, 0].set(xlabel='bucket', ylabel='value')
    axs[0, 0].set_title('F1 (popular class)')
    axs[0, 1].plot(df["bucket"], df["popular_class_prec"])
    axs[0, 1].set(xlabel='bucket', ylabel='value')
    axs[0, 1].set_title('Precision (popular class)')
    axs[0, 2].plot(df["bucket"], df["popular_class_rec"])
    axs[0, 2].set(xlabel='bucket', ylabel='value')
    axs[0, 2].set_title('Recall (popular class)')

    axs[1, 0].plot(df["bucket"], df["unpopular_class_f1"])
    axs[1, 0].set(xlabel='bucket', ylabel='value')
    axs[1, 0].set_title('F1 (unpopular class)')
    axs[1, 1].plot(df["bucket"], df["unpopular_class_prec"])
    axs[1, 1].set(xlabel='bucket', ylabel='value')
    axs[1, 1].set_title('Precision (unpopular class)')
    axs[1, 2].plot(df["bucket"], df["unpopular_class_rec"])
    axs[1, 2].set(xlabel='bucket', ylabel='value')
    axs[1, 2].set_title('Recall (unpopular class)')

    axs[2, 0].plot(df["bucket"], df["weighted_f1"])
    axs[2, 0].set(xlabel='bucket', ylabel='value')
    axs[2, 0].set_title('F1 (weighted)')
    axs[2, 1].plot(df["bucket"], df["weighted_prec"])
    axs[2, 1].set(xlabel='bucket', ylabel='value')
    axs[2, 1].set_title('Precision (weighted)')
    axs[2, 2].plot(df["bucket"], df["weighted_rec"])
    axs[2, 2].set(xlabel='bucket', ylabel='value')
    axs[2, 2].set_title('Recall (weighted)')

    axs[3, 0].plot(df["bucket"], df["f1_macro"])
    axs[3, 0].set(xlabel='bucket', ylabel='value')
    axs[3, 0].set_title('F1 (macro)')
    axs[3, 1].plot(df["bucket"], df["prec_macro"])
    axs[3, 1].set(xlabel='bucket', ylabel='value')
    axs[3, 1].set_title('Precision (macro)')
    axs[3, 2].plot(df["bucket"], df["rec_macro"])
    axs[3, 2].set(xlabel='bucket', ylabel='value')
    axs[3, 2].set_title('Recall (macro)')

    axs[4, 0].plot(df["bucket"], df['auc_macro'])
    axs[4, 0].set(xlabel='bucket', ylabel='value')
    axs[4, 0].set_title('AUC (macro)')

    axs[4, 1].plot(df["bucket"], df['auc_weighted'])
    axs[4, 1].set(xlabel='bucket', ylabel='value')
    axs[4, 1].set_title('AUC (weighted)')

    fig.set_size_inches(18.5, 10.5)
    plt.savefig(output_file_name)

    plt.clf()
    plt.close(fig)