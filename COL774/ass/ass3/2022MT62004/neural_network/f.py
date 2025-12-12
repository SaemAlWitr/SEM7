import pandas as pd
import numpy as np
from sklearn.metrics import *
import argparse
from joblib import Parallel, delayed
from .neural_network import *
import cv2

target_names = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
n_classes = len(target_names)
parser = argparse.ArgumentParser()
parser.add_argument("train",type=str, default="../data_nn_int/train")
parser.add_argument("test",type=str, default="../data_nn_int/test")
parser.add_argument("out", type=str, default="outputs")
args = parser.parse_args()

X_train, X_test, y_train,y_test = load_data(args.train, args.test)

layers = [512,256,128,64]

try:
    model= NeuralNetwork.load('models/f/f-trained-depth-'+str(layers))
except FileNotFoundError:
    def train_for_unit(l, X_train, y_train):
        l = [(i,'relu') for i in l]
        nn = NeuralNetwork(n=3072, layers=l, r=(n_classes, 'softmax'))
        nn.fit(X_train, y_train, lr=1e-2, batch_size=32, tol=1e-4, verbose=3, epochs=20, out_path=f'plots/f/epoch-vs-loss-{len(l)}-layers.png',f1=True, X_test=X_test, y_test=y_test)
        return nn

    model = train_for_unit(layers, X_train, y_train)

    model.save('models/f/f-trained-depth-'+str(layers))

f1_score_arr = model.f1_arr
f1_score_arr_test = model.f1_arr_test
print(f"f1 scores train {f1_score_arr}")
print(f"f1 scores test {f1_score_arr_test}")
fig = plt.figure(figsize=(8,4))
plt.plot(np.arange(len(f1_score_arr))+1, f1_score_arr, color='b',marker='o',label= "F1 score train")
plt.plot(np.arange(len(f1_score_arr_test))+1, f1_score_arr_test, color='r',marker='o',label= "F1 score test")
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.legend()
plt.title('Epoch vs Average F1 score (trained from scratch)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'plots/f/epoch-vs-f1score-{len(layers)}-layers.png',dpi=300, bbox_inches='tight')
plt.close(fig)
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(X_test)

print("classification report for training data")
print(classification_report(y_train, y_train_preds,target_names=target_names))
print("classification report for test data")
print(classification_report(y_test, y_test_preds,target_names=target_names))

f1_train = f1_score(y_train, y_train_preds,average='weighted')
f1_test = f1_score(y_test, y_test_preds,average='weighted')

print(f"avg f1 score train {f1_train}")
print(f"avg f1 score test {f1_test}")
try:
    digit_model = NeuralNetwork.load("models/f/finetuned")
except FileNotFoundError:
    try:
        consonant_model = NeuralNetwork.load_old('models/d/d-trained-depth-'+str(layers))
    except FileNotFoundError:
        raise FileNotFoundError("consonant model not found. do part d")
    digit_model = NeuralNetwork(n = 3072, layers=[(i,'relu') for i in layers], r=(n_classes,'softmax'))
    for i,j in enumerate(consonant_model.layer[:-1]):
        digit_model.layer[i] = np.copy(j)
    digit_model.fit(X_train, y_train, epochs=20, lr=0.01, batch_size=32, tol=1e-4, verbose=3, out_path=f'plots/f/epoch-vs-loss-finetuned.png',f1=True, X_test=X_test, y_test=y_test)
    digit_model.save("models/f/finetuned")

f1_finetuned = digit_model.f1_arr
f1_finetuned_test = digit_model.f1_arr_test


print(f"f1 scores train {f1_finetuned}")
print(f"f1 scores test {f1_finetuned_test}")
fig = plt.figure(figsize=(8,4))
plt.plot(np.arange(len(f1_finetuned))+1, f1_finetuned, color='b',marker='o',label= "F1 score train")
plt.plot(np.arange(len(f1_finetuned))+1, f1_finetuned_test, color='r',marker='o',label= "F1 score test")
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.title('Epoch vs Average F1 score (finetuned)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'plots/f/epoch-vs-f1score-{len(layers)}-finetuned.png',dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"f1 scores train {f1_finetuned}")
print(f"f1 scores test {f1_finetuned_test}")

epochs = np.arange(len(f1_finetuned))+1
fig = plt.figure(figsize=(8,4))
plt.plot(np.arange(len(f1_finetuned))+1, f1_finetuned, color='b',marker='o',label= "F1 score train finetuned")
plt.plot(np.arange(len(f1_finetuned))+1, f1_finetuned_test, color='r',marker='o',label= "F1 score test finetuned")
plt.plot(np.arange(len(f1_score_arr))+1, f1_score_arr, color='y',marker='o',label= "F1 score train scratch")
plt.plot(np.arange(len(f1_score_arr_test))+1, f1_score_arr_test, color='g',marker='o',label= "F1 score test scratch")

plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.title('Model epochs vs F1 score')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
out_path = 'plots/f/epoch-vs-f1score-compare.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print("Saved:", out_path)
plt.close(fig)

y_test_preds = [model.predict(X_test),digit_model.predict(X_test)]

y_preds = []
for i in y_test_preds:
    y_preds.extend(np.argmax(i,axis=1)+37)

submission = pd.DataFrame({'prediction':y_preds})
submission.to_csv(args.out+'/precitions_f.csv', index=False)