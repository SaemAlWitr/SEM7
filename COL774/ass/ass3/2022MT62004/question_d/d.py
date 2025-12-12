import pandas as pd
import numpy as np
from sklearn.metrics import *
import argparse
from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import *

parser = argparse.ArgumentParser()
parser.add_argument("train",type=str, default="../data/train.csv")
parser.add_argument("val",type=str, default="../data/validation.csv")
parser.add_argument("test",type=str, default="../data/test.csv")
parser.add_argument("out", type=str, default="result")
args = parser.parse_args()


cat = ['team', 'opp', 'host', 'month']
num = ['Unnamed','year', 'fow','score','rpo','toss', 'day_match', 'bat_first', 'format']

df_train, df_val, df_test = load_data(args.train, args.val, args.test, cat=cat, num=num)

#collect columns with more than 2 unique vals
col_tobe_encoded = []
for col in cat:
    if np.unique(df_train[col]).size > 2:
        col_tobe_encoded.append(col)

# do one-hot encoding for categorical columns. using pandas
df_train = pd.get_dummies(df_train,columns=col_tobe_encoded)
df_test = pd.get_dummies(df_test, columns=col_tobe_encoded).reindex(columns=df_train.columns, fill_value=0)
df_val = pd.get_dummies(df_val, columns=col_tobe_encoded).reindex(columns=df_train.columns, fill_value=0)

# print(df_train.columns)
num = num
cat = list(set(df_train.columns).difference(num+['result']))

max_depth_values = [15, 25, 35, 45]
models = [DecisionTree(max_depth=i, cat=cat, num=num,criterion='gini') for i in max_depth_values]
# train
models = Parallel(n_jobs=8, verbose = 2)(delayed(models[i].fit)(df_train) for i in range(len(models)))
print("num nodes\ttrain acc\tval acc\ttest acc")
for i, model in enumerate(models):
    n, acc_train, acc_val, acc_test = model.prune(df_train, df_val, df_test, class_col = 'result')
    print(f"{n[-1]}\t{acc_train[-1]}\t{acc_val[-1]}\t{acc_test[-1]}")
    plot_n_vs_acc(n, acc_train, acc_val, acc_test, out_path=f'size-vs-accuracies-d-{max_depth_values[i]}.png')
if 35 in max_depth_values:
    y_pred_test=models[max_depth_values.index(35)].predict(df_test)
    submission = pd.DataFrame({'result':y_pred_test})
    submission.to_csv(args.out,index=False)