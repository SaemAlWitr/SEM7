from sklearn.svm import SVC
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import *
import cvxopt
cvxopt.solvers.options['show_progress'] = False
import os,glob
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
labels = {}
labels_inv = {}
for i, name in enumerate(names):
    labels[name] = i
    labels_inv[i] = name
print(labels)
size = (32,32)
tol = 1e-6
def load_data():
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    train_paths = glob.glob("data/train/"+names[4]+"/*.jpg")+glob.glob("data/train/"+names[5]+"/*.jpg")
    test_paths = glob.glob("data/test/"+names[4]+"/*.jpg")+glob.glob("data/test/"+names[5]+"/*.jpg")
    m_train = len(train_paths)
    m_test = len(test_paths)
    train_images = np.empty((m_train, 3*32*32),dtype=np.float32)
    test_images = np.empty((m_test, 3*32*32),dtype=np.float32)

    for i,file in enumerate(train_paths):
        with Image.open(file) as im:
            im.thumbnail(size)
            arr = np.reshape(im, (3*32*32,)).astype(np.float32)
            arr/=255
            train_images[i] = arr
    for i,file in enumerate(test_paths):
        with Image.open(file) as im:
            im.thumbnail(size)
            arr = np.reshape(im, (3*32*32,)).astype(np.float32)
            arr/=255
            test_images[i] = arr
    
    train_labels = [-1]*int(m_train/2)+[1]*int(m_train/2)
    test_labels = [-1]*int(m_test/2)+[1]*int(m_test/2)
    df_train = pd.DataFrame({'image':list(train_images), 'label':train_labels})
    df_test = pd.DataFrame({'image':list(test_images), 'label':test_labels})

    return df_train, df_test

def load_all_data():
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    train_paths = []
    test_paths = []
    train_labels = []
    test_labels = []
    for i, name in enumerate(names):
        train_paths.extend(glob.glob("data/train/"+name+"/*.jpg"))
        test_paths.extend(glob.glob("data/test/"+name+"/*.jpg"))
        train_labels.extend([i]*len(glob.glob("data/train/"+name+"/*.jpg")))
        test_labels.extend([i]*len(glob.glob("data/test/"+name+"/*.jpg")))
    
    m_train = len(train_paths)
    m_test = len(test_paths)
    train_images = np.empty((m_train, 3*32*32),dtype=np.float32)
    test_images = np.empty((m_test, 3*32*32),dtype=np.float32)

    for i,file in enumerate(train_paths):
        with Image.open(file) as im:
            im.thumbnail(size)
            arr = np.reshape(im, (3*32*32,)).astype(np.float32)
            arr/=255
            train_images[i] = arr
    for i,file in enumerate(test_paths):
        with Image.open(file) as im:
            im.thumbnail(size)
            arr = np.reshape(im, (3*32*32,)).astype(np.float32)
            arr/=255
            test_images[i] = arr
    
    
    df_train = pd.DataFrame({'image':list(train_images), 'label':train_labels})
    df_test = pd.DataFrame({'image':list(test_images), 'label':test_labels})

    return df_train, df_test
        

def fit(df: pd.DataFrame, image_col: str, class_col: str, C = 1.0, kernel = 'linear', gamma = 0.001, tol = 1e-6, t_time = True):
    t0 = time.perf_counter()
    m = df.shape[0]
    Y = df[class_col].to_numpy()
    X = np.empty((m,3*32*32))
    for i, arr in enumerate(df[image_col]):
        X[i] = arr
    if kernel == 'linear':
        K = sklearn.metrics.pairwise.linear_kernel(X,X)
    elif kernel == 'gaussian':
        K = sklearn.metrics.pairwise.rbf_kernel(X,X,gamma=gamma)

    Q = (Y[:, None] * Y[None, :]) * K
    P = cvxopt.matrix(2.0*Q)

    q = cvxopt.matrix(-np.ones((m,1)))
    G = cvxopt.matrix(np.vstack([-np.eye(m),
                                np.eye(m)]).astype(np.float64))
    h = cvxopt.matrix(np.hstack([np.zeros((m,)), C*np.ones((m,))]).astype(np.float64), (2*m,1))
    A = cvxopt.matrix(Y.reshape((1,m)).astype(np.float64))
    b = cvxopt.matrix(0.0)
    alpha = np.array(cvxopt.solvers.qp(P,q,G,h,A,b)['x']).reshape((m,))
    alpha = np.clip(alpha,0.0,C)
    coeff = Y*alpha
    SV_idx = np.where(alpha > tol)[0]
    nSV = SV_idx.shape[0]
    free_idx = np.where((alpha > tol) & (alpha < C-tol))[0]
    if free_idx.size == 0:
        idxs = SV_idx
    else:
        idxs = free_idx

    if idxs.size == 0:
        b = 0.0
    else:
        bis = Y[idxs][:,None] - K[idxs].dot(coeff)
        b = float(np.mean(bis))
    t1 = time.perf_counter()
    if t_time:
        print(f"Training time = {t1-t0:.6f} s")
    return coeff, b, nSV, SV_idx

def predict(df, train_images, image_col, predicted_col, coeff, b, kernel = 'linear', gamma = 0.001):
    m = df.shape[0]
    m_train = coeff.shape[0]
    X = np.empty((m,3*32*32))
    Z = np.empty((m_train,3*32*32))
    for i, arr in enumerate(df[image_col]):
        X[i] = arr
    for i, arr in enumerate(train_images):
        Z[i] = arr
    if kernel=='linear':
        K = sklearn.metrics.pairwise.linear_kernel(X,Z)
    elif kernel=='gaussian':
        K = sklearn.metrics.pairwise.rbf_kernel(X,Z,gamma = gamma)
    y_pred = np.ones((m,))
    pos_pred = np.where(((K.dot(coeff)) + b) < 0)
    y_pred[pos_pred] *= -1
    df[predicted_col] = y_pred

class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self,C = 1, kernel = 'linear', gamma = 0.001, train_fn = None, tol = 1e-6):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.train_fn = train_fn
        self.tol = tol
    def fit(self,X,y):
        classes = np.unique(y)
        self.neg_label = classes[0]
        self.pos_label = classes[1]
        y_mapped = np.where(y == self.pos_label, 1.0, -1.0)
        df = pd.DataFrame({'image':list(X), 'label':y_mapped})
        self.coeff, self.b, self.nSV, _ = self.train_fn(df, 'image', 'label', C = self.C, kernel=self.kernel, gamma=self.gamma, tol = self.tol, t_time = False)
        SV_idxs = np.where(np.abs(self.coeff) > tol)[0]
        if SV_idxs.size != 0:
            self.SVs = X[SV_idxs]
            self.coeff = self.coeff[SV_idxs]
        else:
            self.SVs = X
        return self
    def decision_function(self,X):
        m = X.shape[0]
        if self.kernel=='linear':
            K = sklearn.metrics.pairwise.linear_kernel(X,self.SVs)
        elif self.kernel=='gaussian':
            K = sklearn.metrics.pairwise.rbf_kernel(X, self.SVs, gamma = self.gamma)
        scores = K.dot(self.coeff)+self.b
        return scores
    def predict(self, X):
        scores = self.decision_function(X)
        preds = np.where(scores >= 0, 1.0, -1.0)
        mapped = np.where(preds == 1.0, self.pos_label, self.neg_label)
        return mapped


    

def metric(df, class_col, predicted_col):
    return accuracy_score(df[class_col],df[predicted_col])

def save_model(coeff,b,nSV,kernel, filename = "params.npz"):
    np.savez(filename, coeff = coeff, b = b, nSV = nSV, kernel=kernel)

def load_model(filename = "params.npz"):
    params = np.load(filename, allow_pickle=True)
    return params["coeff"], params["b"], params["nSV"], params["kernel"]

def plot_w_sv_top5(df, image_col, coeff, w = None, C = 1, top_k = 5,tol = 1e-6, kernel = 'linear'):
    m = df.shape[0]
    X = np.empty((m,3*32*32))
    for i, arr in enumerate(df[image_col]):
        X[i] = arr
    alpha = np.abs(coeff)
    SV_idxs = np.where((alpha > tol))[0]
    SV_alpha = alpha[SV_idxs]

    top_idx = np.argsort(-SV_alpha)[:top_k]
    fig, axes = plt.subplots(1, top_k, figsize=(3*top_k, 3))
    for i, idx in enumerate(top_idx):
        idx = SV_idxs[idx]
        img = X[idx].reshape(32,32,3)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"SV idx = {idx}\n alpha[{idx}] = {alpha[idx]:.4f}")
    plt.suptitle("Top-5 support vectors")
    plt.tight_layout()
    out_path = "top5-sv-"+kernel+'-cvx.png'
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    if type(w) != type(None):
        w_img = w.reshape((32,32,3))
        w_max = np.abs(w_img).max()
        if w_max == 0:
            w_norm = w_img
        else:
            w_norm = (w_img) / w_max
        # plt.imshow(w_norm)
        # plt.title("Weight vector")
        # plt.axis('off')
        # plt.show()
        w_gray = w_norm.mean(axis=2)

        fig = plt.figure(figsize=(5,5))
        im = plt.imshow(w_gray, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Weight vector w (channel-mean, normalized, diverging colormap)")
        plt.axis('off')
        out_path = "w-"+kernel+'-cvx.png'
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

def get_w(df, image_col, coeff):
    m = df.shape[0]
    X = np.empty((m,3*32*32))
    for i, arr in enumerate(df[image_col]):
        X[i] = arr
    return X.T @ coeff

# df_test = pd.read_pickle("binary-class-predicted.pkl")
df_train, df_test = load_data()
m_train = df_train.shape[0]
m_test = df_test.shape[0]
X_train= np.empty((m_train,3*32*32))
for i, arr in enumerate(df_train['image']):
    X_train[i] = arr
X_test = np.empty((m_test,3*32*32))
for i, arr in enumerate(df_test['image']):
    X_test[i] = arr
print(X_train.shape)
# linear kernel
print("linear-kernel CVXOPT")

coeff,b,nSV,lin_sv_idx = fit(df_train, 'image', 'label',tol=tol, kernel='linear')
# save_model(coeff,b,nSV,'linear',"params.npz")
# coeff,b,nSV,kernel = load_model("params.npz")
print(f"nSV = {nSV}, nSV% = {nSV/m_train*100:.4f}%")

predict(df_test,df_train['image'], 'image', 'predicted cvxopt linear', coeff, b,kernel='linear')
print(f"test accuracy: {metric(df_test, 'label', 'predicted cvxopt linear')*100:.4f}")
coeff_lin, b_lin = coeff, b
plot_w_sv_top5(df_train, 'image', coeff, get_w(df_train, 'image', coeff),tol = tol,kernel='linear')

# # gaussian kernel
print("gaussian kernel CVXOPT")
kernel = 'gaussian'

coeff,b,nSV,gauss_sv_idx = fit(df_train, 'image', 'label',kernel=kernel, tol = tol)

# save_model(coeff,b,nSV,kernel,"params-gauss.npz")
# coeff,b,nSV,kernel = load_model("params-gauss.npz")
print(f"nSV = {nSV}, nSV% = {nSV/m_train*100:.4f}%")

predict(df_test,df_train['image'], 'image', 'predicted cvxopt gaussian', coeff, b,kernel=kernel)
print(f"test accuracy: {metric(df_test, 'label', 'predicted cvxopt gaussian')*100:.4f}%")

plot_w_sv_top5(df_train, 'image', coeff,tol=tol,kernel='gaussian')

# # linear-kernel sklearn LIBSVM
print("linear-kernel sklearn LIBSVM")

clf = SVC(kernel='linear', C=1.0,tol=tol)
t0 = time.perf_counter()
clf.fit(X_train, df_train['label'])
t1 = time.perf_counter()
print(f"Training time = {t1-t0:.6f} s")
df_test['predicted libsvm linear'] = clf.predict(X_test)  
nSV =  clf.n_support_.sum()
print(f"nSV = {nSV}, nSV% = {nSV/m_train*100:.4f}%")
coeff, b = load_model("params.npz")[0:2]
print(f"squared error btw LIBSVM w and CVXOPT w = {np.linalg.norm(clf.coef_-get_w(df_train, 'image', coeff_lin))}")
print(f"absolute difference in LIBSVM b and CVXOPT b = {np.abs(b_lin-clf.intercept_[0])}")
print(f"test accuracy: {metric(df_test, 'label', 'predicted libsvm linear')*100:.4f}%")
svs = clf.support_
print(f"# Same support vectors={len(set(svs).intersection(lin_sv_idx))}")
alpha = np.abs(clf.dual_coef_[0])
top_idx = np.argsort(-alpha)[:5]
fig, axes = plt.subplots(1, 5, figsize=(3*5, 3))
for i, _idx in enumerate(top_idx):
    idx = svs[_idx]
    img = np.array(X_train[idx]).reshape(32,32,3)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"SV idx = {idx}\n alpha[{idx}] = {alpha[_idx]:.4f}")
plt.suptitle("Top-5 support vectors")
plt.tight_layout()
out_path = "top5-sv-linear-lib.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
w = clf.coef_
w_img = w.reshape((32,32,3))
w_max = np.abs(w_img).max()
if w_max == 0:
    w_norm = w_img
else:
    w_norm = (w_img) / w_max
# plt.imshow(w_norm)
# plt.title("Weight vector")
# plt.axis('off')
# plt.show()
w_gray = w_norm.mean(axis=2)

fig = plt.figure(figsize=(5,5))
im = plt.imshow(w_gray, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Weight vector w (channel-mean, normalized, diverging colormap)")
plt.axis('off')
out_path = 'w-linear-lib.png'
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)


# # rbf kernel
print("gaussian kernel sklearn LIBSVM")

clf = SVC(kernel='rbf', C=1.0,gamma=0.001,tol=tol)
t0 = time.perf_counter()
clf.fit(X_train, df_train['label'])
t1 = time.perf_counter()
print(f"Training time = {t1-t0:.6f} s")
df_test['predicted libsvm gaussian'] = clf.predict(X_test)
nSV = clf.n_support_.sum()
print(f"nSV = {nSV}, nSV% = {nSV/m_train*100:.4f}%")
print(f"test accuracy: {metric(df_test, 'label', 'predicted libsvm gaussian')*100:.4f}%")
svs = clf.support_
print(f"# Same support vectors={len(set(svs).intersection(gauss_sv_idx))}")
alpha = np.array(clf.dual_coef_[0])
top_idx = np.argsort(-alpha)[:5]
fig, axes = plt.subplots(1, 5, figsize=(3*5, 3))
for i, _idx in enumerate(top_idx):
    idx = svs[_idx]
    img = X_train[idx].reshape(32,32,3)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"SV idx = {idx}\n alpha[{idx}] = {alpha[_idx]:.4f}")
plt.suptitle("Top-5 support vectors")
plt.tight_layout()
out_path = "top5-sv-gaussian-lib.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

df_test.to_pickle("binary-class-predicted.pkl")

# one vs one 45 models CVXOPT 10 classes
print("gaussian kernel CVXOPT one-vs-one")

df_train, df_test = load_all_data()
m_train = df_train.shape[0]
m_test = df_test.shape[0]

X_train= np.empty((m_train,3*32*32))
for i, arr in enumerate(df_train['image']):
    X_train[i] = arr

X_test = np.empty((m_test,3*32*32))
for i, arr in enumerate(df_test['image']):
    X_test[i] = arr
y_train = df_train['label'].to_numpy()
y_test = df_test['label'].to_numpy()

base = SVM(C=1.0, kernel='gaussian', gamma=0.001, train_fn=fit)
ovo = OneVsOneClassifier(base,n_jobs=-1)

t0 = time.perf_counter()
ovo.fit(X_train, y_train)
t1 = time.perf_counter()
t = int(t1-t0)
print(f"Training time = {t//60}min {t%60}s")

df_test['Predicted CVXOPT ovo gaussian'] = ovo.predict(X_test)

print(f"test accuracy: {metric(df_test, 'label', 'Predicted CVXOPT ovo gaussian')*100:.4f}%")

# one vs one 45 models libsvm 10 classes linear
print("linear kernel LIBSVM one-vs-one")

ovo = OneVsOneClassifier(SVC(kernel='linear', C = 1.0, tol = tol), n_jobs=-1)

t0 = time.perf_counter()
ovo.fit(X_train, y_train)
t1 = time.perf_counter()
t = int(t1-t0)
print(f"Training time = {t//60}min {t%60}s")

df_test['Predicted LIBSVM ovo linear'] = ovo.predict(X_test)

print(f"test accuracy: {metric(df_test, 'label', 'Predicted LIBSVM ovo linear')*100:.4f}%")


# one vs one 45 models libsvm 10 classes gaussian
print("gaussian kernel LIBSVM one-vs-one")

ovo = OneVsOneClassifier(SVC(kernel='rbf', C = 1.0, gamma=0.001, tol = tol), n_jobs=-1)

t0 = time.perf_counter()
ovo.fit(X_train, y_train)
t1 = time.perf_counter()
t = int(t1-t0)
print(f"Training time = {t//60}min {t%60}s")
df_test['Predicted LIBSVM ovo gaussian'] = ovo.predict(X_test)

print(f"test accuracy: {metric(df_test, 'label', 'Predicted LIBSVM ovo gaussian')*100:.4f}%")


df_test.to_pickle("ovo-predicted.pkl")
# df_test = pd.read_pickle("ovo-predicted.pkl")
confusion_matrix_L = confusion_matrix(df_test['label'], df_test['Predicted LIBSVM ovo gaussian'])
confusion_matrix_C = confusion_matrix(df_test['label'], df_test['Predicted CVXOPT ovo gaussian'])

disp1 = ConfusionMatrixDisplay(confusion_matrix_C, display_labels=np.arange(10))
disp2 = ConfusionMatrixDisplay(confusion_matrix_L, display_labels=np.arange(10))

fig, axes = plt.subplots(1,2,figsize = (10, 5))
disp1.plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Confusion Matrix - CVXOPT")
disp2.plot(ax=axes[1], cmap='Greens')
axes[1].set_title("Confusion Matrix - LIBSVM")
plt.tight_layout()
out_path = "cm-gaussian-ovo-cvx-lib.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
print("Saved:", out_path)
# plt.show()    
plt.close(fig)

misclassified = np.where(df_test['Predicted CVXOPT ovo gaussian'] != df_test['label'])[0]
np.random.shuffle(misclassified)
misclassified_10 = misclassified[:10]
fig, axes = plt.subplots(2, 5, figsize = (4,3))
for i, img in enumerate(df_test['image'][misclassified_10]):
    img = np.reshape(img, (32,32,3))
    axes[0 if i<5 else 1, i%5].imshow(img)
    axes[0 if i<5 else 1, i%5].axis('off')
    axes[0 if i<5 else 1, i%5].set_title(f"actual: {labels_inv[df_test['label'][misclassified_10[i]]]}\n pred:{labels_inv[df_test['Predicted CVXOPT ovo gaussian'][misclassified_10[i]]]}", fontsize = 7)
plt.suptitle("10 misclassified examples")
plt.tight_layout()
out_path = "misclassified_examples.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
print("Saved:", out_path)
# plt.show()    
plt.close(fig)

'''
- the most easily classified classes by the model are automobile, frog, ship
- automobile and truck are similar which is understandable
- airplane and ship are confusing mostly because of their backgrouds being similar (blue sky and blue water)
- bird is being recognised as a deer or frog again due to the background (barks and trees) also, some birds like ostrich have big legs
can be confusing
- cat is being recognised as a frog and a dog. frogs have wide face, some cats have too. dogs and cats are home animals, backgroud 
- deer is being confused with bird and frog as discussed above
- dog is being confused with a cat, both have the same background of home
- frog is being confused with a deer in some places bg again
- horse is being confused with a deer, similar backgrounds
- ship and airplane were confusing background based again and both have the same metal panels kind of
- truck ~ automobile
'''
df_train, df_test = load_all_data()
m_train = df_train.shape[0]
m_test = df_test.shape[0]

X_train= np.empty((m_train,3*32*32))
for i, arr in enumerate(df_train['image']):
    X_train[i] = arr

X_test = np.empty((m_test,3*32*32))
for i, arr in enumerate(df_test['image']):
    X_test[i] = arr
y_train = df_train['label'].to_numpy()
y_test = df_test['label'].to_numpy()
# k-fold cross-validation for C in {1e-5, 1e-3, 1, 5, 10}
print("k-fold cross-validation for C in {1e-5, 1e-3, 1, 5, 10}")
C_values = [1e-5, 1e-3, 1, 5, 10]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"C":C_values}
grid = GridSearchCV(
    estimator= SVC(kernel='rbf', gamma=0.001),
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2.1
)
t0 = time.perf_counter()
grid.fit(X_train, y_train)
t1 = time.perf_counter()
t = int(t1-t0)
print(f"GridSearchCV time = {t//60}min {t%60}s")
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
stds = grid.cv_results_['std_test_score']
print('Validation Accuracies')
print("C\tmean_val_acc\tstd_val_acc")
for param, mean, std in zip(params, means, stds):
    print(f"{param}\t{mean:.4f}\t{std}")
test_means = []

print('Test Accuracies\nC\tmean_test_acc')
for C in C_values:
    clf = OneVsOneClassifier(SVC(kernel='rbf', C=C, gamma=0.001), n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_means.append(acc)
    print(f"{C}\t{acc:.4f}")

Cs = np.array(C_values)
val_means = means
test_means = test_means

plt.figure(figsize=(7,4))
plt.plot(Cs, val_means, marker='o', linestyle='-', label='Validation mean')
plt.plot(Cs, test_means, marker='s', linestyle='--', label='Test mean')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
out_path = "C_vs_accuracies-cv-test.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print("Saved:", out_path)
plt.show()

best_C = C_values[np.argmax(val_means)]

print(f"Best C={best_C}, validation accuracy={val_means.max()}")

clf = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.001, C=best_C), n_jobs=-1)
clf.fit(X_train, y_train)
df_test["best C Predicted"] = y_pred = clf.predict(X_test)
df_test.to_pickle("best-c-predicted.pkl")
print(f"Using C={best_C}, test accuracy={accuracy_score(y_test,y_pred):.4f}")