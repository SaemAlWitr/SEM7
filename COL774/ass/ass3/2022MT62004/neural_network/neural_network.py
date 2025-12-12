import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
import glob, cv2
import re
import numpy as np
import json
import os,sys

def load_data(train_path, test_path):
    train_folders = glob.glob(train_path+'/*')
    train_folders.sort()
    test_folders = glob.glob(test_path+'/*')
    test_folders.sort()
    n_classes = len(train_folders)
    n_train_examples_perclass = [0]*len(train_folders)
    n_test_examples_perclass = [0]*len(train_folders)
    train_images_path = []
    test_images_path = []
    for i,folder in enumerate(train_folders):
        paths = glob.glob(folder+'/*')
        n_train_examples_perclass[i] = len(paths)
        train_images_path.extend(paths)
    for i,folder in enumerate(test_folders):
        paths = glob.glob(folder+'/*')
        n_test_examples_perclass[i] = len(paths)
        test_images_path.extend(paths)
    def sort_key(p):
        match = re.search(r'/(\d+)/(\d+)\.png$', p)
        folder, file_num = match.groups()
        return int(folder), int(file_num)
    train_images_path = sorted(train_images_path, key=sort_key)
    test_images_path = sorted(test_images_path, key=sort_key)
    X_train = np.zeros((np.sum(n_train_examples_perclass,dtype=int),3072))
    X_test = np.zeros((np.sum(n_test_examples_perclass,dtype=int),3072))
    y_train = np.zeros((np.sum(n_train_examples_perclass,dtype=int), n_classes),dtype=np.int8)
    y_test = np.zeros((np.sum(n_test_examples_perclass,dtype=int), n_classes),dtype=np.int8)
    for i, filename in enumerate(train_images_path):
        img = cv2.imread(filename, cv2.IMREAD_COLOR).reshape((3072,)).astype(np.float32)
        X_train[i,:] = img/255
    for i, filename in enumerate(test_images_path):
        img = cv2.imread(filename, cv2.IMREAD_COLOR).reshape((3072,)).astype(np.float32)
        X_test[i,:] = img/255
    y_train[np.arange(np.sum(n_train_examples_perclass,dtype=int)), np.repeat(np.arange(n_classes),n_train_examples_perclass)] = 1
    y_test[np.arange(np.sum(n_test_examples_perclass,dtype=int)), np.repeat(np.arange(n_classes),n_test_examples_perclass)] = 1
    
    return X_train, X_test, y_train,y_test 

class NeuralNetwork:
    def __init__(self, n = 1, layers = [(1, 'relu')], r = (1, 'softmax'), random_state=42):
        self.n = n
        self.names = {'relu':0, 'softmax':1, 'sigmoid':2}
        self.described = [(i, self.names[j]) for i, j in layers]
        self.described.append((r[0], self.names[r[1]]))
        self.r = r[0]
        self.activation=None
        self.random_state = random_state
        self.rng = np.random.mtrand._rand if random_state is None else np.random.RandomState(random_state)
        self.layer = []
        prev = self.n
        for i,act in self.described:
            # using silimar initialization as MLPClass.. helps keep Var(z)=Var(theta.x) = var(x)
            bound = np.sqrt(6/(prev+i))
            self.layer.append(self.rng.uniform(-bound, bound, (i, prev)).astype(np.float32))
            prev = i
        self.activation = [i[1] for i in self.described]
        self.b =  [np.zeros((i.shape[0],)) for i in self.layer]
        self.loss_arr = []
        self.f1_arr = []
        self.f1_arr_test = []
    def save(self, path):
        path = path + '.npz'

        meta = {
            'n': self.n,
            'described': [list(t) for t in self.described],
            'activation': self.activation,
            'r': self.r,
            'random_state':self.random_state,
            'loss_arr':self.loss_arr,
            'f1_arr':self.f1_arr,
            'f1_arr_test':self.f1_arr_test
        }
        meta_json = json.dumps(meta)

        save_dict = {'meta': meta_json}
        for idx, w in enumerate(self.layer):
            save_dict[f'layer_{idx}'] = np.asarray(w)
            save_dict[f'b_{idx}'] = np.asarray(self.b[idx])
        try:
            np.savez_compressed(path, **save_dict)
        except Exception as e:
            print("unable to save")

    @classmethod
    def load(cls, path):
        path = path + '.npz'
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")

        npz = np.load(path, allow_pickle=True)
        meta_str = npz['meta'].tolist() if isinstance(npz['meta'], np.ndarray) else npz['meta']
        meta = json.loads(meta_str)

        layer_keys = [k for k in npz.files if k.startswith('layer_')]
        layer_indices = sorted([int(k.split('_', 1)[1]) for k in layer_keys])

        layers = [np.array(npz[f'layer_{i}'], copy=True) for i in layer_indices]
        bs     = [np.array(npz[f'b_{i}'], copy=True) for i in layer_indices]

        for w in layers:
            w.setflags(write=True)
        for b in bs:
            b.setflags(write=True)

        inst = object.__new__(cls)

        inst.n = meta['n']
        inst.names = {'relu':0, 'softmax':1, 'sigmoid':2}
        inst.described = [tuple(x) for x in meta['described']]
        inst.r = meta['r']
        inst.activation = meta['activation']
        inst.layer = layers
        inst.b = bs
        inst.loss_arr=meta['loss_arr']
        inst.f1_arr=meta['f1_arr']
        inst.f1_arr_test=meta['f1_arr_test']
        random_state = meta['random_state']
        inst.random_state = random_state
        inst.rng = np.random.mtrand._rand if random_state is None else np.random.RandomState(random_state)
        return inst
    @classmethod
    def load_old(cls, path):
        path = path + '.npz'
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")

        npz = np.load(path, allow_pickle=True)
        meta_str = npz['meta'].tolist() if isinstance(npz['meta'], np.ndarray) else npz['meta']
        meta = json.loads(meta_str)

        layer_keys = [k for k in npz.files if k.startswith('layer_')]
        layer_indices = sorted([int(k.split('_', 1)[1]) for k in layer_keys])

        layers = [np.array(npz[f'layer_{i}'], copy=True) for i in layer_indices]
        bs     = [np.array(npz[f'b_{i}'], copy=True) for i in layer_indices]

        for w in layers:
            w.setflags(write=True)
        for b in bs:
            b.setflags(write=True)

        inst = object.__new__(cls)

        inst.n = meta['n']
        inst.names = {'relu':0, 'softmax':1, 'sigmoid':2}
        inst.described = [tuple(x) for x in meta['described']]
        inst.r = meta['r']
        inst.activation = meta['activation']
        inst.layer = layers
        inst.b = bs
        random_state = meta['random_state']
        inst.random_state = random_state
        inst.rng = np.random.mtrand._rand if random_state is None else np.random.RandomState(random_state)
        return inst

    def g(self, net, act, out):
        if act == 0:
            return np.maximum(0, net, out=out)    
        if act == 1:
            np.exp(net-np.max(net,axis=1,keepdims=True), out=out)
            out/=out.sum(axis=1,keepdims=True)
        if act == 2:
            np.exp(net,out=out)
            out/=(1+out)

    
    def g_(self, o, act, eps = 1e-12):
        if act == 0:
            return (o > eps).astype(np.float32)
        if act == 1 or act == 2:
            return o*(1-o)
    
    def decision_function(self, X, out):
        np.matmul(X, self.layer[0].T, out=out[0])
        self.g(out[0]+self.b[0], self.activation[0], out[0])
        for i in range(1, len(out)):
            np.matmul(out[i-1], self.layer[i].T, out=out[i])
            self.g(out[i]+self.b[i], self.activation[i], out[i])
        return out

    def loss(self, X, y):
        probs = self.predict_proba(X)
        J = -np.sum(np.log(probs+1e-12)*y)/X.shape[0]
        return J
    
    def predict_proba(self, X):
        out = [np.zeros((X.shape[0], i.shape[0])) for i in self.layer]
        self.decision_function(X, out=out)
        return out[-1]
    
    def predict(self,X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        one_hot = np.zeros(probs.shape, dtype=probs.dtype)
        one_hot[np.arange(X.shape[0]), idx] = 1
        return one_hot

    def fit(self, X, y,epochs=350, lr = 1e-4, batch_size = 32, tol = 1e-3, verbose = 0, n_iter_no_change = 10, out_path = 'plots/epoch-vs-loss.png', f1 = False, X_test = None, y_test=None):
        m = X.shape[0]
        der_wrt_theta = [np.zeros(i.shape) for i in self.layer]
        der_wrt_b = [np.zeros_like(i) for i in self.b]
        _out = [np.zeros((batch_size, i.shape[0])) for i in self.layer]
        _der_wrt_net = [np.zeros((batch_size,i.shape[0])) for i in self.layer]
        l = len(self.layer)
        loss = self.loss(X,y)
        best_loss = loss
        loss_arr = [loss]
        if f1: 
            f1_score_arr = [f1_score(y, self.predict(X), average='weighted')]
            f1_score_arr_test = [f1_score(y_test, self.predict(X_test), average='weighted')]
        consequtive_bad = 0
        if verbose:
            print(f"initial loss {loss:.6f}")
        sys.stdout.flush()

        for epoch in range(epochs):
            perm = self.rng.permutation(m)
            X_shuff = X[perm]
            y_shuff = y[perm]
            for start_idx in range(0,m,batch_size):
                m_batch = min(m-start_idx, batch_size)
                end_idx = start_idx+m_batch
                out = _out if m_batch == batch_size else [i[:m_batch,:] for i in _out]
                der_wrt_net = _der_wrt_net if m_batch == batch_size else [i[:m_batch,:] for i in _der_wrt_net]
                
                #forward propagate
                X_batch = X_shuff[start_idx:end_idx]
                y_batch = y_shuff[start_idx:end_idx]
                self.decision_function(X_batch, out=out)
                der_wrt_net[-1] = (out[-1]-y_batch)/m_batch # should be /m_batch (m, n^L)
                # der_wrt_theta[-1] = der_wrt_net[-1].T @ (out[-2] if l > 1 else X_batch)
                np.dot(der_wrt_net[-1].T, (out[-2] if l > 1 else X_batch), out=der_wrt_theta[-1])
                np.mean(der_wrt_net[-1],axis=0,out=der_wrt_b[-1])
                for i in range(l-2,-1,-1):
                    theta = self.layer[i+1]
                    g_ = self.g_(out[i], act = self.activation[i])
                    np.dot(der_wrt_net[i+1],theta,out= der_wrt_net[i])
                    np.multiply(der_wrt_net[i],g_,out=der_wrt_net[i]) # (m,n^i)
                    np.mean(der_wrt_net[i],axis=0,out=der_wrt_b[i])
                    if i!=0:
                        np.dot(der_wrt_net[i].T,out[i-1],out=der_wrt_theta[i])
                    else:
                        np.dot(der_wrt_net[i].T , X_batch,out=der_wrt_theta[i])
                    
                for i in range(l):
                    self.layer[i]-=lr*der_wrt_theta[i]
                    self.b[i]-=lr*der_wrt_b[i]
            new_loss = self.loss(X,y)
            if verbose > 1 and (epoch % 10 == 0 or epoch == epochs - 1):
                if verbose == 2: print(f"epoch {epoch:4d} loss {new_loss:.6f} n_units {self.layer[0].shape[0]}")
                if verbose == 3: print(f"epoch {epoch:4d} loss {new_loss:.6f} depth {len(self.layer)-1}")
            loss = new_loss
            loss_arr.append(loss)
            if f1: 
                f1_score_arr.append(f1_score(y, self.predict(X), average='weighted'))
                f1_score_arr_test.append(f1_score(y_test, self.predict(X_test), average='weighted'))
            if loss < best_loss-tol:
                best_loss = loss
                consequtive_bad=0
            else:
                consequtive_bad+=1
            if consequtive_bad >= n_iter_no_change:
                break
        if verbose :
            fig = plt.figure()
            plt.plot(np.arange(len(loss_arr)), loss_arr)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Epoch vs Loss')
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(out_path,dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"final loss {loss:.6f}")
        sys.stdout.flush()
        self.loss_arr = loss_arr
        if f1: 
            self.f1_arr = f1_score_arr
            self.f1_arr_test = f1_score_arr_test
        return self



