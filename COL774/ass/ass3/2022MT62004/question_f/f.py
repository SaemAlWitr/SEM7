import pandas as pd
import numpy as np
from sklearn.metrics import *
import argparse
from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
parser = argparse.ArgumentParser()
parser.add_argument("train",type=str, default="../data/train.csv")
parser.add_argument("val",type=str, default="../data/validation.csv")
parser.add_argument("test",type=str, default="../data/test.csv")
parser.add_argument("out", type=str, default="result")
args = parser.parse_args()


cat = ['team', 'opp', 'host', 'month', 'toss', 'day_match', 'bat_first', 'format']
num = ['year', 'fow','score','rpo']

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

X_train = df_train.drop(columns=['result']).to_numpy()
y_train = df_train['result'].to_numpy()
X_val = df_val.drop(columns=['result']).to_numpy()
y_val = df_val['result'].to_numpy()
X_test = df_test.drop(columns=['result']).to_numpy()
y_test = df_test['result'].to_numpy()

X_combined = np.concatenate([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

test_fold = np.concatenate([
    -1 * np.ones(len(X_train), dtype=int),
     0 * np.ones(len(X_val), dtype=int)
])

custom_cv = PredefinedSplit(test_fold)

param_grid = {
    'n_estimators':[50, 150, 250, 350],
    'max_features':[0.1,0.3,0.5,0.7,0.9],
    'min_samples_split':[2,4,6,8,10],
    'ccp_alpha':[0.0001,0.0003,0.0005,0.0007]
}

model = GridSearchCV(
    RandomForestClassifier(
        criterion='entropy',
        oob_score=True,
        bootstrap=True,
        random_state=42
        ),
    param_grid=param_grid,
    cv=custom_cv,
    scoring='accuracy',
    verbose=2,
    n_jobs=7,
)

model.fit(X_combined, y_combined)

best_rf = model.best_estimator_

acc_train = accuracy_score(y_train, best_rf.predict(X_train))
acc_val = accuracy_score(y_val, best_rf.predict(X_val))
acc_test = accuracy_score(y_test, best_rf.predict(X_test))
acc_obb = best_rf.oob_score_

print("Best Parameters:", model.best_params_)
print(f"Training Accuracy:\t{acc_train:.4f}")
print(f"OOB Accuracy:\t{acc_obb:.4f}")
print(f"Validation Accuracy:\t{acc_val:.4f}")
print(f"Test Accuracy:\t{acc_test:.4f}")

y_pred_test=model.best_estimator_.predict(df_test)
submission = pd.DataFrame({'result':y_pred_test})
submission.to_csv(args.out,index=False)