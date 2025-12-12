import pandas as pd
import numpy as np
from sklearn.metrics import *
import argparse
from joblib import Parallel, delayed
from neural_network import *
import cv2

target_names = ['क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
    'क्ष', 'त्र', 'ज्ञ',]
n_classes = len(target_names)
parser = argparse.ArgumentParser()
parser.add_argument("train",type=str, default="../data_nn/train")
parser.add_argument("test",type=str, default="../data_nn/test")
parser.add_argument("out", type=str, default="outputs")
args = parser.parse_args()

X_train, X_test, y_train,y_test = load_data(args.train, args.test)

units = [1,5,10,50,100]

try:
    models = [NeuralNetwork.load_old('models/b/b-trained-n_units-'+str(i)) for i in units]
except FileNotFoundError:
    def train_for_unit(n_units, X_train, y_train):
        nn = NeuralNetwork(n=3072, layers=[(n_units, 'sigmoid')], r=(n_classes, 'softmax'))
        nn.fit(X_train, y_train, lr=1e-2, batch_size=32, tol=1e-5, verbose=2, epochs=1000, out_path=f'plots/b/epoch-vs-loss-{n_units}-units.png')
        return nn

    models = Parallel(n_jobs=-1, verbose=2)(
        delayed(train_for_unit)(u, X_train, y_train) for u in units
    )

    [models[i].save('models/b/b-trained-n_units-'+str(units[i])) for i in range(len(units))]

y_train_preds = Parallel(n_jobs=-1, verbose=2)(delayed(models[i].predict)(X_train) for i in range(len(units)))
y_test_preds = Parallel(n_jobs=-1, verbose=2)(delayed(models[i].predict)(X_test) for i in range(len(units)))

print("classification report for training data")
for i in range(len(units)):
    print(f"n_units={units[i]}")
    print(classification_report(y_train, y_train_preds[i],target_names=target_names))
print("classification report for test data")
for i in range(len(units)):
    print(f"n_units={units[i]}")
    print(classification_report(y_test, y_test_preds[i],target_names=target_names))

f1_train = np.array([f1_score(y_train, y_train_preds[i],average='weighted') for i in range(len(units))])
f1_test = np.array([f1_score(y_test, y_test_preds[i],average='weighted') for i in range(len(units))])

fig = plt.figure(figsize=(7,4))
plt.plot(units, f1_train, marker='o', linestyle='-', color='b', label='F1 score Train')
plt.plot(units, f1_test, marker='s', linestyle='--', color='r', label='F1 score Test')
plt.scatter(units, f1_train.astype(np.float16), color='b')
plt.scatter(units, f1_test.astype(np.float16), color='r')

for xi, yi in zip(units, np.round(f1_train,decimals=4)):
    plt.annotate(f'({xi}, {yi})', (xi, yi),
                 textcoords="offset points", xytext=(5,5), ha='right',)
for xi, yi in zip(units, np.round(f1_test,decimals=4)):
    plt.annotate(f'({xi}, {yi})', (xi, yi),
                 textcoords="offset points", xytext=(5,5), ha='left')
plt.xlabel('No. of units in Hidden Layer')
plt.ylabel('F1 score')
plt.title('No. of units in Hidden Layer vs F1 score')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
out_path = 'plots/b/n-units-vs-f1score.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print("Saved:", out_path)
plt.close(fig)
y_preds = []
for i in y_test_preds:
    y_preds.extend(np.argmax(i,axis=1)+1)

submission = pd.DataFrame({'prediction':y_preds})
submission.to_csv(args.out+'/precitions_b.csv', index=False)
