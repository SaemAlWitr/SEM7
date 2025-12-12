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
models = [DecisionTree(max_depth=i, cat=cat, num=num) for i in max_depth_values]
# train
models = Parallel(n_jobs=-1, verbose = 2)(delayed(models[i].fit)(df_train) for i in range(len(models)))
#predict
y_preds_train = [models[i].predict(df_train) for i in range(len(max_depth_values))]
y_preds_val = [models[i].predict(df_val) for i in range(len(max_depth_values))]
y_preds = [models[i].predict(df_test) for i in range(len(max_depth_values))]
#get accs
accs_train = np.array([accuracy_score(df_train['result'], y_preds_train[i]) for i in range(len(max_depth_values))])
accs_val = np.array([accuracy_score(df_val['result'], y_preds_val[i]) for i in range(len(max_depth_values))])
accs = np.array([accuracy_score(df_test['result'], y_preds[i]) for i in range(len(max_depth_values))])
#print accs
for i in range(len(max_depth_values)):
    if(i == 0):
        print("max_depth\ttrain accuracy\tval accuracy\ttest accuracy")
    print(f"{max_depth_values[i]}\t{accs_train[i]}\t{accs_val[i]}\t{accs[i]}")
#plot accs
plot_d_vs_acc(max_depth_values, accs_train, accs,accs_val=accs_val, out_path="max-d-vs-accuracies.png")
#print best acc
best = np.argmax(accs_val)
print(f"best max_depth value={max_depth_values[best]} with test accuracy={accs[best]}")
#save result for best acc
best_model = models[best]
y_pred_test = best_model.predict(df_test, 'predicted')
acc_test = accuracy_score(df_test['result'], y_pred_test)

submission = pd.DataFrame({'result':y_pred_test})
submission.to_csv(args.out, index=False)

