import pandas as pd
import numpy as np
from sklearn.metrics import *
import argparse
from joblib import Parallel, delayed
import joblib
from .neural_network import *
import cv2
from sklearn.neural_network import MLPClassifier
import json, sys
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

layers = [[512],[512,256],[512,256,128],[512,256,128,64]]

try:
    models = []
    for l in layers:
        path = 'models/e/e-trained-depth-'+str(l)+'.pkl'
        models.append(joblib.load(path))
except FileNotFoundError:
    def train_for_unit(l, X_train, y_train):
        nn = MLPClassifier(hidden_layer_sizes=l,random_state=42, activation='relu', solver='sgd', alpha=0, batch_size=32, learning_rate='constant', learning_rate_init=0.01, max_iter=400, verbose=True,shuffle=True)
        nn.fit(X_train, y_train)
        sys.stdout.flush()
        joblib.dump(nn,'models/e/e-trained-depth-'+str(l)+'.pkl')
        return nn

    models = Parallel(n_jobs=-1, verbose=2)(
        delayed(train_for_unit)(i, X_train, y_train) for i in layers
    )

for i,nn in enumerate(models):
    fig = plt.figure(figsize=(7,4))
    losses = nn.loss_curve_
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path = 'plots/e/epoch-vs-loss'+str(len(layers[i]))+'.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Saved:", out_path)
    plt.close(fig)


y_train_preds = Parallel(n_jobs=-1, verbose=2)(delayed(models[i].predict)(X_train) for i in range(len(layers)))
y_test_preds = Parallel(n_jobs=-1, verbose=2)(delayed(models[i].predict)(X_test) for i in range(len(layers)))

print("classification report for training data")
for i in range(len(layers)):
    print(f"units per layer={layers[i]}")
    print(classification_report(y_train, y_train_preds[i],target_names=target_names))
print("classification report for test data")
for i in range(len(layers)):
    print(f"units per layer={layers[i]}")
    print(classification_report(y_test, y_test_preds[i],target_names=target_names))

f1_train = np.array([f1_score(y_train, y_train_preds[i],average='weighted') for i in range(len(layers))])
f1_test = np.array([f1_score(y_test, y_test_preds[i],average='weighted') for i in range(len(layers))])

fig = plt.figure(figsize=(7,4))
depths = [len(i) for i in layers]
plt.plot(depths, f1_train, marker='o', linestyle='-', color='b', label='F1 score Train')
plt.plot(depths, f1_test, marker='s', linestyle='--', color='r', label='F1 score Test')
plt.scatter(depths, f1_train.astype(np.float16), color='b')
plt.scatter(depths, f1_test.astype(np.float16), color='r')

for xi, yi in zip(depths, np.round(f1_train,decimals=4)):
    plt.annotate(f'({xi}, {yi})', (xi, yi),
                 textcoords="offset points", xytext=(5,5), ha='right',)
for xi, yi in zip(depths, np.round(f1_test,decimals=4)):
    plt.annotate(f'({xi}, {yi})', (xi, yi),
                 textcoords="offset points", xytext=(5,5), ha='left')
plt.xlabel('Model Depth')
plt.ylabel('F1 score')
plt.title('Model Depths vs F1 score')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
out_path = 'plots/e/depth-vs-f1score.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print("Saved:", out_path)
plt.close(fig)

y_preds = []
for i in y_test_preds:
    y_preds.extend(np.argmax(i,axis=1)+1)

submission = pd.DataFrame({'prediction':y_preds})
submission.to_csv(args.out+'/precitions_e.csv', index=False)