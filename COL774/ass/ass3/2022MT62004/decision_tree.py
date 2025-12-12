import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import *
def load_data(train_path, val_path, test_path, cat = None, num = None):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    for col in cat:
        df_test[col] = pd.Categorical(df_test[col], dtype='category')
        df_train[col] = pd.Categorical(df_train[col], dtype='category')
        df_val[col] = pd.Categorical(df_val[col], dtype='category')
    return df_train, df_val, df_test


class DecisionTree:
    def __init__(self, max_depth = 10, criterion = 'entropy', tol = 1e-6, cat = None, num = None):
        self.max_depth = max_depth
        assert(criterion == 'entropy' or criterion=='gini')
        self.criterion = criterion
        self.tol = tol
        self.cat = cat
        self.num = num
        self.root = None
    def entropy(self, p):
        p = p[p>0]
        return -np.dot(p, np.log2(p))
    def gini(self,p):
        return 1-np.sum(np.power(p, 2))
    def fit(self, df:pd.DataFrame, class_col='result'):
        n=0
        training_idx = df.index.to_numpy()
        classes = np.unique(df[class_col],)
        p = df.loc[training_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
        if p.sum() == 0:
            return
        p/=np.sum(p)
        self.root = {
            "score":self.entropy(p) if self.criterion =='entropy' else self.gini(p),
            "pred":classes[np.argmax(p)],
            "depth":0,
            "par":(None, None),
            "size":1,
            "children":{},
            "rule":None,
            'id':n
        }
        n+=1
        q = deque([self.root])
        q_idx = deque([training_idx])
        while(len(q)):
            node = q.popleft()
            training_idx = q_idx.popleft()
            if node['depth'] >= self.max_depth:
                break
            if training_idx.size < 2:
                continue
            # if node.entropy <self.tol or node.entropy>1-self.tol:
            #     continue
            max_mi = -np.inf
            rule = None
            for col in self.cat:

                cats = np.unique(df[col])
                
                priors = np.zeros((cats.size,),dtype=np.float64)
                h = np.zeros((cats.size,),dtype=np.float64)
                for i,category in enumerate(cats):
                    cat_idx = training_idx[(df.loc[training_idx, col].to_numpy() == category)]
                    p = df.loc[cat_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                    if p.sum() == 0:
                        continue
                    p/=np.sum(p)
                    h[i] = self.entropy(p) if self.criterion=='entropy' else self.gini(p)
                    priors[i] = cat_idx.size
                if priors.sum() == 0:
                    continue
                priors/=np.sum(priors)
                E_entropy = np.dot(priors, h)
                curr_mi = node["score"] - E_entropy
                if curr_mi > max_mi:
                    max_mi = curr_mi
                    rule = (col,)
            for col in self.num:
                med = df.loc[training_idx, col].median()
                left_idx = training_idx[df.loc[training_idx, col].to_numpy() <= med]
                right_idx = training_idx[df.loc[training_idx, col].to_numpy() > med]
                left_n = left_idx.size
                right_n = right_idx.size
                if left_n ==0 or right_n==0:
                    continue
                priors = np.array([left_n,right_n],dtype=np.float64)/(left_n+ right_n)
                h = [0,0]
                left_p = df.loc[left_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                left_p/=left_p.sum()
                h[0] = self.entropy(left_p) if self.criterion=='entropy' else self.gini(left_p)
                right_p = df.loc[right_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                right_p/=right_p.sum()
                h[1] = self.entropy(right_p) if self.criterion=='entropy' else self.gini(right_p)
                E_entropy = np.dot(priors, h)
                curr_mi = node["score"] - E_entropy
                if curr_mi > max_mi:
                    max_mi = curr_mi
                    rule = (col,med)
            if rule is None or max_mi <= self.tol:
                continue
            node["rule"] = rule
            children = dict()
            if len(rule) == 1:
                col = rule[0]
                cats = np.unique(df[col])
                # perform split and write to 2queues (make sure to handle cases where child is empty or child is pure) and create child nodes add chid nodes to node 
                for category in cats:
                    new_training_idx = training_idx[(df.loc[training_idx, col].to_numpy() == category)]
                    if new_training_idx.size == 0:
                        continue
                    p = df.loc[new_training_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                    p/=np.sum(p)
                    children[category]={
                        "score":self.entropy(p) if self.criterion =='entropy' else self.gini(p),
                        "pred":classes[np.argmax(p)],
                        "depth":node["depth"]+1,
                        "par":(node, category),
                        "size":1,
                        "children":{},
                        "rule":None,
                        'id':n
                    }
                    n+=1
                    q.append(children[category])
                    q_idx.append(new_training_idx)
                node["children"] = children
            else:
                med = rule[1]
                col = rule[0]
                left_idx = training_idx[df.loc[training_idx, col].to_numpy() <= med]
                right_idx = training_idx[df.loc[training_idx, col].to_numpy() > med]
                left_n = left_idx.size
                right_n = right_idx.size
                if left_n ==0 or right_n==0:
                    continue
                # left
                p = df.loc[left_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                if left_idx.size == 0:
                    continue
                p/=np.sum(p)
                children['l']={
                    "score":self.entropy(p) if self.criterion =='entropy' else self.gini(p),
                    "pred":classes[np.argmax(p)],
                    "depth":node["depth"]+1,
                    "par":(node, 'l'),
                    "size":1,
                    "children":{},
                    "rule":None,
                    'id':n
                }
                n+=1
                q.append(children['l'])
                q_idx.append(left_idx)                
                # right
                p = df.loc[right_idx, class_col].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float64)
                if right_idx.size == 0:
                    continue
                p/=np.sum(p)
                children['r']={
                    "score":self.entropy(p) if self.criterion =='entropy' else self.gini(p),
                    "pred":classes[np.argmax(p)],
                    "depth":node["depth"]+1,
                    "par":(node, 'r'),
                    "size":1,
                    "children":{},
                    "rule":None,
                    'id':n
                }
                n+=1
                q.append(children['r'])
                q_idx.append(right_idx) 
                node["children"]= children 
        self.num_nodes()
        return self             
    def num_nodes(self):
        if self.root is None:
            return 0
        if self.root['size'] == 1:
            stack = [self.root]
            vis = set()
            while len(stack):
                node = stack[-1]
                for child in node['children'].values():
                    if child['id'] not in vis:
                        stack.append(child)
                        break
                else:
                    stack.pop()
                    vis.add(node['id'])
                    ct = 1
                    for child in node['children'].values():
                        ct+=child['size']
                    node['size'] = ct
        return self.root['size']
    
    def predict(self, df:pd.DataFrame, predicted_col=None):
        if self.root is None:
            raise ValueError("Tree is empty. Train first")
        y_pred = np.empty(df.shape[0], dtype = np.int8)
        for i in range(df.shape[0]):
            x = df.iloc[i,:]
            node = self.root
            while node['rule'] != None:
                if len(node['rule']) == 1:
                    col = node['rule'][0]
                    if x[col] in node['children']:
                        node = node['children'][x[col]]
                    else:
                        break
                else:
                    col, med = node['rule']
                    if x[col] <= med:
                        if 'l' in node['children']:
                            node = node['children']['l']
                        else:
                            break
                    else:
                        if 'r' in node['children']:
                            node = node['children']['r']
                        else:
                            break
            y_pred[i] = node['pred']
        if predicted_col is not None:
            df[predicted_col] = y_pred 
        return y_pred
    def prune_(self, df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, class_col = 'result'):
        if self.root is None:
            raise ValueError("Tree is empty. Train first")
        n = [self.root['size']]
        acc_train = [accuracy_score(df_train[class_col], self.predict(df_train))]
        acc_val = [accuracy_score(df_val[class_col], self.predict(df_val ))]
        acc_test = [accuracy_score(df_test[class_col], self.predict(df_test ))]
        changed = True
        vis = np.zeros((self.root['size'],),dtype=bool)
        while changed:
            changed = False
            stack = [self.root]
            vis[:] = False
            best_acc = acc_val[-1]
            best_node = self.root
            # print(n[-1])
            while len(stack):
                node = stack[-1]
                for child in node['children'].values():
                    if not vis[child['id']]:
                        stack.append(child)
                        break
                else:
                    vis[node['id']]=True
                    stack.pop()
                    par, cat = node['par']
                    if par is None:
                        continue
                    par['children'].pop(cat)
                    accuracy_val = accuracy_score(df_val[class_col], self.predict(df_val))
                    if accuracy_val >= best_acc:
                        best_acc = accuracy_val
                        best_node = node
                    par['children'][cat] = node
            if best_acc > acc_val[-1]:
                changed = True
                par,cat = best_node['par']
                if par is None:
                    break
                par['children'].pop(cat)
                while par is not None:
                    par['size']-=best_node['size']
                    par,_ = par['par']
                
                acc_train.append(accuracy_score(df_train[class_col], self.predict(df_train )))
                acc_val.append(best_acc)
                acc_test.append(accuracy_score(df_test[class_col], self.predict(df_test )))
                n.append(self.root['size'])
                        
        return [n, acc_train, acc_val, acc_test]
    def prune(self, df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, class_col='result'):

        if self.root is None:
            raise ValueError("Tree is empty. Train first")
        classes = np.unique(df_train[class_col].to_numpy())
        class_inv = {c: i for i, c in enumerate(classes)}

        def init_node_counts(node):
            node['train_cts'] = np.zeros((len(classes),),dtype=np.int32)
            node['val_cts'] = np.zeros((len(classes),), dtype=np.int32)
            node['test_cts'] = np.zeros((len(classes),),dtype=np.int32)
            node['train_leaf_cor'] = 0 
            node['val_leaf_cor'] = 0
            node['test_leaf_cor']  = 0
            for child in node['children'].values():
                init_node_counts(child)
        init_node_counts(self.root)

        def route_dataset(df, counts_attr):
            n = df.shape[0]
            for i in range(n):
                row = df.iloc[i]
                true = row[class_col]
                idx = class_inv[true]
                node = self.root
                node[counts_attr][idx] += 1
                while node['rule'] is not None:
                    rule = node['rule']
                    if len(rule) == 1:
                        col = rule[0]
                        val = row[col]
                        if val in node['children']:
                            node = node['children'][val]
                            node[counts_attr][idx] += 1
                        else:
                            break
                    else:
                        col, med = rule
                        if row[col] <= med:
                            if 'l' in node['children']:
                                node = node['children']['l']
                                node[counts_attr][idx] += 1
                            else:
                                break
                        else:
                            if 'r' in node['children']:
                                node = node['children']['r']
                                node[counts_attr][idx] += 1
                            else:
                                break

        route_dataset(df_train,'train_cts')
        route_dataset(df_val, 'val_cts')
        route_dataset(df_test,'test_cts')

        def compute_leaf_corr(node):
            if not node['children']:
                pred = node['pred']
                pred_idx = class_inv[pred]
                node['train_leaf_cor'] = int(node['train_cts'][pred_idx])
                node['val_leaf_cor'] = int(node['val_cts'][pred_idx])
                node['test_leaf_cor'] = int(node['test_cts'][pred_idx])
                return (node['train_leaf_cor'],node['val_leaf_cor'], node['test_leaf_cor'])

            tr = va = te = 0
            for child in node['children'].values():
                cr, cv, ct = compute_leaf_corr(child)
                tr += cr
                va += cv
                te += ct
            node['train_leaf_cor'] = int(tr)
            node['val_leaf_cor'] = int(va)
            node['test_leaf_cor'] = int(te)
            return (tr, va, te)

        total_tr, total_val, total_test = compute_leaf_corr(self.root)
        N_train = df_train.shape[0]
        N_val = df_val.shape[0]
        N_test= df_test.shape[0]

        n_list = [self.root['size']]
        acc_train = [total_tr/N_train]
        acc_val   = [total_val/N_val]
        acc_test  = [total_test/N_test]

        def collect_nodes_postorder():
            stack = [self.root]
            order = []
            while stack:
                node = stack.pop()
                order.append(node)
                for child in node['children'].values():
                    stack.append(child)
            return list(reversed(order))

        improved = 1
        while improved:
            improved = 0
            delta_best = 0
            best_node = None

            postorder = collect_nodes_postorder()
            for node in postorder:
                if not node['children']:
                    continue
                par, key = node['par']
                if par is None:
                    continue
                pred = node['pred']
                pred_idx = class_inv[pred]
                new_corr_val = int(node['val_cts'][pred_idx])
                old_corr_val = int(node['val_leaf_cor'])
                delta = new_corr_val - old_corr_val

                if delta > delta_best:
                    delta_best = delta
                    best_node = node
            if best_node is not None and delta_best > 0:
                node = best_node
                par, key = node['par']

                pred_idx = class_inv[node['pred']]
                new_tr = int(node['train_cts'][pred_idx])
                new_val = int(node['val_cts'][pred_idx])
                new_te = int(node['test_cts'][pred_idx])

                old_tr = int(node['train_leaf_cor'])
                old_val = int(node['val_leaf_cor'])
                old_te = int(node['test_leaf_cor'])

                delta_tr = new_tr - old_tr
                delta = new_val - old_val
                delta_te = new_te - old_te

                removed_size = node['size'] - 1
                node['children'] = {}
                node['rule'] = None
                node['size'] = 1

                node['train_leaf_cor'] = new_tr
                node['val_leaf_cor'] = new_val
                node['test_leaf_cor'] = new_te

                anc = par
                while anc is not None:
                    anc['train_leaf_cor'] += delta_tr
                    anc['val_leaf_cor'] += delta
                    anc['test_leaf_cor'] += delta_te
                    anc['size'] -= removed_size
                    anc, _ = anc['par']

                total_tr += delta_tr
                total_val += delta
                total_test += delta_te

                acc_train.append(total_tr/N_train)
                acc_val.append(total_val/N_val)
                acc_test.append(total_test/N_test)
                n_list.append(self.root['size'])

                improved=1

        return [n_list, acc_train, acc_val, acc_test]
    
def plot_d_vs_acc(max_depth_values, accs_train = [], accs = [], accs_val = [], out_path = "out.png"):
    fig = plt.figure(figsize=(7,4))
    plt.plot(max_depth_values, accs_train, marker='o', linestyle='-', label='Train accuracies')
    if len(accs_val):
        plt.plot(max_depth_values, accs_val, marker='s', linestyle='--', label='Validation accuracies')
    plt.plot(max_depth_values, accs, marker='s', linestyle='--', label='Test accuracies')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Max Depth vs Accuracy')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Saved:", out_path)
    plt.close(fig)

def plot_n_vs_acc(n_values, acc_train = [], acc_val = [], acc_test = [], out_path = "out.png"):
    fig = plt.figure(figsize=(7,4))
    if len(acc_train):
        plt.plot(n_values, acc_train, label='Train accuracies')
    if len(acc_val):
        plt.plot(n_values, acc_val, label='Validation accuracies')
    if len(acc_test):
        plt.plot(n_values, acc_test, label='Test accuracies')
    plt.xlabel('Tree Size')
    plt.ylabel('Accuracy')
    plt.title('Tree Size vs Accuracy')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Saved:", out_path)
    plt.close(fig)