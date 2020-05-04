DATA

It was employed the train data of FOLD1.

RESULTS

Summary of results [LINEAR REGRESSION]

    Actual     Predicted
0        0 -4.884181e+07
1        1  8.145623e+06
2        0  1.079781e+07
3        1 -2.310273e+07
4        0  3.754552e+07
5        0 -4.289214e+06
6        1  6.988966e+06
7        0  5.392679e+06
8        0 -2.034758e+07
9        0  2.166298e+07
10       0  2.804398e+07
11       1 -2.237834e+07
12       2 -1.099401e+07
13       0  3.485450e+06
14       2 -8.666412e+06
15       2  8.909488e+05
16       2 -1.844772e+06
17       1  3.034339e+06
18       2 -2.115423e+07
19       0  2.923933e+06
20       0 -5.576761e+07
21       0  1.515343e+07
22       0  4.377544e+07
23       0 -2.190413e+07
24       2  2.896557e+07

Evaluation results [LINEAR REGRESSION]

Mean Absolute Error (MAE): 27042100.456406195
Mean Squared Error (MSE): 3.184783000488081e+16
Root Mean Squared Error (RMSE): 178459603.28567585
Spearman correlation: SpearmanrResult(correlation=-0.10286859411154027, pvalue=0.0)
NDCG score:  0.8664693616694245

------------------------------------------------------------------------------------

[LightGBM] [Info] Total Bins 2968
[LightGBM] [Info] Number of data: 413118, number of used features: 15
[LightGBM] [Info] Start training from score 0.678186

Summary of results [LIGHT GBM]

    Actual  Predicted
0        0   0.656616
1        1   0.682010
2        0   0.702619
3        1   0.688516
4        0   0.635889
5        0   0.668084
6        1   0.690706
7        0   0.627553
8        0   0.686636
9        0   0.664848
10       0   0.673596
11       1   0.687345
12       2   0.685903
13       0   0.709910
14       2   0.675173
15       2   0.654059
16       2   0.712109
17       1   0.656542
18       2   0.682106
19       0   0.662230
20       0   0.694556
21       0   0.687678
22       0   0.665842
23       0   0.698781
24       2   0.627768

Evaluation results [LIGHT GBM]

Mean Absolute Error (MAE): 0.690847785394287
Mean Squared Error (MSE): 0.6711589292994014
Root Mean Squared Error (RMSE): 0.8192429000604163
Spearman correlation: SpearmanrResult(correlation=0.20098447722377355, pvalue=0.0)
NDCG score:  0.9083823181462107

------------------------------------------------------------------------------------

[LAMBDA RANK]

It is given an error with the whole data. It must be explored why.

