import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import *
import time



class NaiveBayes:
    def __init__(self):
        self.m = 0
        self.K = 0
        self.V = 0
        self.vocabulary = {}
        self.log_priors = 0
        self.log_theta = 0

    
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 0-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """
        m = df.shape[0]
        K = df[class_col].max()+1
        # the priors
        priors = np.bincount(df[class_col], minlength=K).astype(np.float16)
        priors /= m
        # flatten the tokens for creating a vocabulary
        flat_tokens = []
        for tokens in df[text_col]:
            if tokens:
                flat_tokens.extend(tokens)
        # create the vocabulary. each word is provided an index which will be it's 
        # position in self.log_theta's colums
        unique_tokens = list(set(flat_tokens))
        self.vocabulary = {token: idx for idx, token in enumerate(unique_tokens)}
        V = len(self.vocabulary)
        # extra for words not in vocabulary
        self.vocabulary[None] = V
        V += 1

        X_indices = []
        for tokens in df[text_col]:
            idxs = [self.vocabulary[token] for token in tokens]
            X_indices.append(np.array(idxs, dtype=np.int32))

        word_counts = np.zeros((K,V))
        total_words = np.zeros((K,))
        Y = df[class_col].to_numpy()
        # create word_counts and total word_counts
        for c in range(K):
            doc_inds = np.nonzero(Y == c)[0]
            if doc_inds.size == 0:
                continue
            arrays = [X_indices[i] for i in doc_inds if X_indices[i].size > 0]
            if len(arrays) == 0:
                continue
            concatenated = np.concatenate(arrays)
            counts = np.bincount(concatenated, minlength=V)
            word_counts[c, :] = counts
            total_words[c] = counts.sum()

        # taking log for numeric stability
        self.log_theta = (np.log(word_counts+smoothening,dtype=np.float32) - np.log(total_words[:,None]+smoothening*V,dtype=np.float32))
        self.log_priors = np.log(priors+1e-12,dtype=np.float32)
        self.K = K
        self.V = V
        return self.log_theta, self.log_priors
    
    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        m_test = int(df.shape[0])
        predictions = np.zeros((m_test,), dtype=np.int32)
        batch_size=500
        V = int(self.V)
        K = int(self.K)
        log_theta_T = self.log_theta.T
        log_priors = self.log_priors
        tmp = np.empty((20,self.K), dtype=np.float32)
        # do in batches so that memory limit is not hit
        for start_idx in range(0, m_test, batch_size):
            end_idx = min(start_idx + batch_size, m_test)
            batch_size_actual = end_idx - start_idx

            batch_token_lists = df[text_col].iloc[start_idx:end_idx].tolist()

            rows = []
            cols = []
            for doc_i, tokens in enumerate(batch_token_lists):
                if not tokens:
                    continue
                mapped = [self.vocabulary.get(t, self.vocabulary.get(None)) for t in tokens]
                cols.extend(mapped)
                rows.extend([doc_i] * len(mapped))

            rows_arr = np.fromiter(rows, dtype=np.int64)
            cols_arr = np.fromiter(cols, dtype=np.int64)
            flat_idx = rows_arr * V + cols_arr

            counts_flat = np.bincount(flat_idx, minlength=batch_size_actual * V)
            word_counts_batch = counts_flat.reshape(batch_size_actual, V).astype(np.float32, copy=False)
            scores = word_counts_batch.dot(log_theta_T)
            scores += log_priors

            batch_preds = np.argmax(scores, axis=1).astype(np.int32)
            predictions[start_idx:end_idx] = batch_preds

            if end_idx % (10 * batch_size) == 0:
                print(f"Processed {end_idx}/{m_test}")

        df[predicted_col] = predictions

def predict_combined(df, K, text_col, title_col, predicted_col, text_vocab, title_vocab, text_log_theta, title_log_theta, log_priors, batch_size = 500):
    '''
    predict function for when title and content have separate thetas and we want a combined model
    '''
    m_test = int(df.shape[0])
    predictions = np.zeros((m_test,), dtype=np.int32)

    text_V = len(text_vocab)
    title_V = len(title_vocab)
    text_log_theta_T = text_log_theta.T
    title_log_theta_T = title_log_theta.T

    for start_idx in range(0, m_test, batch_size):
        end_idx = min(start_idx + batch_size, m_test)
        batch_size_actual = end_idx - start_idx

        batch_text_token_lists = df[text_col].iloc[start_idx:end_idx].tolist()
        batch_title_token_lists = df[title_col].iloc[start_idx:end_idx].tolist()

        rows = []
        cols = []
        for doc_i, tokens in enumerate(batch_text_token_lists):
            if not tokens:
                continue
            mapped = [text_vocab.get(t, text_vocab[None]) for t in tokens]
            cols.extend(mapped)
            rows.extend([doc_i] * len(mapped))

        rows_arr = np.fromiter(rows, dtype=np.int64)
        cols_arr = np.fromiter(cols, dtype=np.int64)
        flat_idx = rows_arr * text_V + cols_arr

        counts_flat = np.bincount(flat_idx, minlength=batch_size_actual * text_V)
        word_counts_batch = counts_flat.reshape(batch_size_actual, text_V).astype(np.float32, copy=False)

        scores = word_counts_batch.dot(text_log_theta_T)

        rows = []
        cols = []
        for doc_i, tokens in enumerate(batch_title_token_lists):
            if not tokens:
                continue
            mapped = [title_vocab.get(t, title_vocab[None]) for t in tokens]
            cols.extend(mapped)
            rows.extend([doc_i] * len(mapped))

        rows_arr = np.fromiter(rows, dtype=np.int64)
        cols_arr = np.fromiter(cols, dtype=np.int64)
        flat_idx = rows_arr * title_V + cols_arr

        counts_flat = np.bincount(flat_idx, minlength=batch_size_actual * title_V)
        word_counts_batch = counts_flat.reshape(batch_size_actual, title_V).astype(np.float32, copy=False)

        scores += word_counts_batch.dot(title_log_theta_T)
        

        scores += log_priors

        batch_preds = np.argmax(scores, axis=1).astype(np.int32)
        predictions[start_idx:end_idx] = batch_preds

        if start_idx % (10 * batch_size) == 0:
            print(f"Processed {end_idx}/{m_test}")

    df[predicted_col] = predictions

def tokenize(df, text_col = "Description", tokenized_col = "Tokenized Description", keep_punctution = True):
    '''
    Tokenizes the given column
    '''
    punctutions = string.punctuation
    if keep_punctution:
        df[tokenized_col] = [nltk.tokenize.word_tokenize(text.lower()) for text in df[text_col]]
    else:
        X = []
        for text in df[text_col]:
            l = nltk.tokenize.word_tokenize(text.lower())
            tokens = []
            for token in l:
                if token not in punctutions:
                    tokens.append(token)
            X.append(tokens)
        df[tokenized_col] = X


def remove_stopwords(df, text_col = "Tokenized Description", removed_stopwords = "Tokenized no stopwords"):
    '''
    removes stopwords. nltk.corpus.stopwords.words('english') is used
    '''
    stop_words = set(nltk.corpus.stopwords.words('english'))
    X = []
    for tokens in df[text_col]:
        x = []
        for token in tokens:
            if token not in stop_words:
                x.append(token)
        X.append(x)
    df[removed_stopwords] = X

def stem(df, text_col = "Tokenized Description", stem_col = "Tokenized stemmed"):
    '''Snowball stemmer'''
    stemmer = nltk.stem.SnowballStemmer('english')
    df[stem_col] = [[stemmer.stem(token) for token in tokens] for tokens in df[text_col]]

def get_bigrams(df, text_col = "Tokenized Description", bigram_col = "Bigrams"):
    df[bigram_col] = [list(nltk.ngrams(tokens,2)) for tokens in df[text_col]]

def save_model(model, filename = "params.npz"):
    np.savez(filename, log_priors = model.log_priors, log_theta = model.log_theta, 
             m = model.m, K = model.K, V = model.V, vocabulary = model.vocabulary)

def load_model(model, filename = "params.npz"):
    params = np.load(filename, allow_pickle=True)
    model.log_priors, model.log_theta, model.m, model.K, model.V, model.vocabulary = params["log_priors"], params["log_theta"], params["m"], params["K"], params["V"], params["vocabulary"].item()

def get_wordcloud(df, text_col = "Tokenized content", class_col = "Class Index", max_words = 200, filename = 'word-clouds.png'):
    K = np.max(df[class_col])+1
    text = ""
    clouds = [WordCloud(background_color='white', max_words=max_words,stopwords=[]) for i in range(K)]
    for c in range(K):
        text = ""
        contents = df.iloc[np.where(df[class_col] == c)][text_col]
        for tokens in contents:
            text += " ".join(tokens)+" "
        clouds[c].generate(text)
    for i, cloud in enumerate(clouds):
        fig = plt.figure()
        plt.imshow(cloud, interpolation='bilinear')
        plt.title(f"Class {i}")
        plt.axis('off')
        out_path = str(i)+filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print("Saved:", out_path)
        plt.close(fig)
    # plt.show(block = True)

def metrics(df_train, df_test, class_col, predicted_col):
    accuracy_train = accuracy_score(df_train[class_col], df_train[predicted_col])
    accuracy_test = accuracy_score(df_test[class_col], df_test[predicted_col])
    print(f"training accuracy: {accuracy_train*100:.4f}% and test accuracy: {accuracy_test*100:.4f}%")


def downloads():
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('punkt')

downloads()

df_train = pd.read_csv("train.csv")
df_train['label'] = df_train['label'].astype(np.int8)
m_train = df_train.shape[0]
df_test = pd.read_csv("test.csv")
df_test['label'] = df_test['label'].astype(np.int8)
m_test = df_test.shape[0]

class_col, content, tokenized, basic_predicted, tokenized_no_sw, tokenized_no_sw_stemmed = 'label', 'content', 'tokenized content', 'basic model predictions', 'tokenized no stopwords', 'tokenized stemmed no stopwords'
stem_no_sw_predicted = 'stemmed no stopwords predicted'
bigram_predicted = 'bigram predicted'
print(m_test, m_train)
print(df_train.columns) # see the column names
#1a train on content and report accuracy over train and test data
# tokenize train and test data. train the model on train data's tokenized column. 
# get accuracy on train and test data
print("tokenizing training data")
tokenize(df_train, text_col=content, tokenized_col=tokenized)
print("tokenizing test data")
tokenize(df_test, text_col=content, tokenized_col=tokenized)

model = NaiveBayes()
t0 = time.perf_counter()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)
print("done")

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=basic_predicted)
t1 = time.perf_counter()
print(f'time={(t1-t0)/2}')
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=basic_predicted)
print("done")
# print("saving databases")
# df_train.to_pickle("train-basic-predicted.pkl")
# df_test.to_pickle("test-basic-predicted.pkl")

# df_train = pd.read_pickle("train-basic-predicted.pkl")
# df_test = pd.read_pickle("test-basic-predicted.pkl")

print("Accuracies without doing any feature engineering/data modefications")
metrics(df_train, df_test, class_col, basic_predicted)

#1b word-cloud
# print("creating word clouds")
get_wordcloud(df_train, text_col=tokenized,class_col=class_col, max_words=200, filename='word-clouds.png')

# 2a stemming and remove stopwords. tokenize, remove stopwords, then stem
# tokenized content col exists. 
print("removing stopwords")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
tokenize(df_train, text_col=content, tokenized_col=tokenized)
tokenize(df_test, text_col=content, tokenized_col=tokenized)
remove_stopwords(df_train, tokenized, tokenized)
remove_stopwords(df_test, tokenized, tokenized)

print("stemming")
stem(df_train, tokenized, tokenized)
stem(df_test, tokenized, tokenized)

# df_train.to_pickle("train-no-sw-stem.pkl")
# df_test.to_pickle("test-no-sw-stem.pkl")

# df_train = pd.read_pickle("train-no-sw-stem.pkl")
# df_test = pd.read_pickle("test-no-sw-stem.pkl")


model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=stem_no_sw_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=stem_no_sw_predicted)
print("done")

# df_train.to_pickle("train-stem-no-sw-predicted.pkl")
# df_test.to_pickle("test-stem-no-sw-predicted.pkl")


df_train = pd.read_pickle("train-stem-no-sw-predicted.pkl")
df_test = pd.read_pickle("test-stem-no-sw-predicted.pkl")


print("Accuracies after stemming (snowball) and stopwords removal")
metrics(df_train, df_test, class_col, stem_no_sw_predicted)

get_wordcloud(df_train, tokenized, class_col = class_col,max_words=200, filename='word-clouds-sw-stem.png')

# after stemming and removing stop-words, with bigrams
bigram_tokenized = 'bigram tokenized content'
get_bigrams(df_train, tokenized,bigram_col=bigram_tokenized)
get_bigrams(df_test, tokenized,bigram_col=bigram_tokenized)

df_train[tokenized] += df_train[bigram_tokenized]
df_test[tokenized] += df_test[bigram_tokenized]

# df_train.to_pickle("train-bigrams.pkl")
# df_test.to_pickle("test-bigrams.pkl")

# df_train = pd.read_pickle("train-bigrams.pkl")
# df_test = pd.read_pickle("test-bigrams.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")

# df_train.to_pickle("train-bigrams-predicted.pkl")
# df_test.to_pickle("test-bigrams-predicted.pkl")

# df_train = pd.read_pickle("train-bigrams-predicted.pkl")
# df_test = pd.read_pickle("test-bigrams-predicted.pkl")


print("Accuracies using bigrams and unigrams with above preprocessing")
metrics(df_train, df_test, class_col, bigram_predicted)

# use bigrams without pre processiong from part 2

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
tokenize(df_train, text_col=content, tokenized_col=tokenized)
tokenize(df_test, text_col=content, tokenized_col=tokenized)

# df_train = pd.read_pickle("train-basic.pkl")
# df_test = pd.read_pickle("test-basic.pkl")

bigram_tokenized = 'bigram tokenized content'
get_bigrams(df_train, tokenized,bigram_col=bigram_tokenized)
get_bigrams(df_test, tokenized,bigram_col=bigram_tokenized)

df_train[tokenized] += df_train[bigram_tokenized]
df_test[tokenized] += df_test[bigram_tokenized]

# df_train.to_pickle("train-bigrams--2.pkl")
# df_test.to_pickle("test-bigrams--2.pkl")

# # print("lodaing tokenized datasets with bigrmas")
# # df_train = pd.read_pickle("train-bigrams--2.pkl")
# # df_test = pd.read_pickle("test-bigrams--2.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")
# df_train.to_pickle("train-bigrams--2-predicted.pkl")
# df_test.to_pickle("test-bigrams--2-predicted.pkl")

# df_train = pd.read_pickle("train-bigrams--2-predicted.pkl")
# df_test = pd.read_pickle("test-bigrams--2-predicted.pkl")


print("Accuracies using bigrams and unigrams without above preprocessing")
metrics(df_train, df_test, class_col, bigram_predicted)

# use bigrams with stopword removal

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
tokenize(df_train, text_col=content, tokenized_col=tokenized)
tokenize(df_test, text_col=content, tokenized_col=tokenized)

# df_train = pd.read_pickle("train-basic.pkl")
# df_test = pd.read_pickle("test-basic.pkl")

remove_stopwords(df_train, tokenized, tokenized)
remove_stopwords(df_test, tokenized, tokenized)

bigram_tokenized = 'bigram tokenized content'
get_bigrams(df_train, tokenized,bigram_col=bigram_tokenized)
get_bigrams(df_test, tokenized,bigram_col=bigram_tokenized)

df_train[tokenized] += df_train[bigram_tokenized]
df_test[tokenized] += df_test[bigram_tokenized]

# df_train.to_pickle("train-bigrams-sw.pkl")
# df_test.to_pickle("test-bigrams-sw.pkl")

# print("lodaing tokenized datasets with bigrmas")
# df_train = pd.read_pickle("train-bigrams-sw.pkl")
# df_test = pd.read_pickle("test-bigrams-sw.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")
print("saving databases")
# df_train.to_pickle("train-bigrams-sw-predicted.pkl")
# df_test.to_pickle("test-bigrams-sw-predicted.pkl")

# df_train = pd.read_pickle("train-bigrams-sw-predicted.pkl")
# df_test = pd.read_pickle("test-bigrams-sw-predicted.pkl")

print("Accuracies using bigrams and unigrams with stopword removal")
metrics(df_train, df_test, class_col, bigram_predicted)

# 5 model based on titles

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
title = 'title'
tokenize(df_train, title, tokenized)
tokenize(df_test, title, tokenized)

# df_train.to_pickle("train-title.pkl")
# df_test.to_pickle("test-title.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=basic_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=basic_predicted)
print("done")
# print("saving databases")
# df_train.to_pickle("train-title-predicted.pkl")
# df_test.to_pickle("test-title-predicted.pkl")

# df_train = pd.read_pickle("train-title-predicted.pkl")
get_wordcloud(df_train, text_col=tokenized,class_col=class_col, max_words=200, filename='word-clouds-title.png')

# df_test = pd.read_pickle("test-title-predicted.pkl")

print("Accuracies using title column")
metrics(df_train, df_test, class_col, basic_predicted)

# title with stem and stopword removal

# df_train = pd.read_pickle("train-title.pkl")
# df_test = pd.read_pickle("test-title.pkl")

remove_stopwords(df_train, tokenized, tokenized)
remove_stopwords(df_test, tokenized, tokenized)

stem(df_train, tokenized, tokenized)
stem(df_test, tokenized, tokenized)

# df_train.to_pickle("train-title-stem-sw.pkl")
# df_train.to_pickle("test-title-stem-sw.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=basic_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=basic_predicted)
print("done")
# print("saving databases")
# df_train.to_pickle("train-title-stem-sw-predicted.pkl")
# df_test.to_pickle("test-title-stem-sw-predicted.pkl")

# df_train = pd.read_pickle("train-title-stem-sw-predicted.pkl")
get_wordcloud(df_train, text_col=tokenized,class_col=class_col, max_words=200, filename='word-clouds-title-sw.png')

# df_test = pd.read_pickle("test-title-stem-sw-predicted.pkl")

print("Accuracies using title column with stemming and stopwords removed")
metrics(df_train, df_test, class_col, basic_predicted)

# title with bigrams and above pre-processing

# df_train = pd.read_pickle("train-title-stem-sw-predicted.pkl")
# df_test = pd.read_pickle("test-title-stem-sw-predicted.pkl")

get_bigrams(df_train, tokenized, 'bigram col')
get_bigrams(df_test, tokenized, 'bigram col')

df_train[tokenized] = df_train[tokenized]+df_train['bigram col']
df_test[tokenized] = df_test[tokenized]+df_test['bigram col']

# df_train.to_pickle("train-title-bigram.pkl")
# df_train.to_pickle("test-title-bigram.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")

# df_train.to_pickle("train-title-bigram-predicted.pkl")
# df_test.to_pickle("test-title-bigram-predicted.pkl")

# df_train = pd.read_pickle("train-title-bigram-predicted.pkl")
# df_test = pd.read_pickle("test-title-bigram-predicted.pkl")

print("Accuracies using title column with stemming and stopwords removed and bigrams")
metrics(df_train, df_test, class_col, bigram_predicted)

# title with bigrams and no preprocessing

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
title = 'title'
tokenize(df_train, title, tokenized)
tokenize(df_test, title, tokenized)
# df_train = pd.read_pickle("train-title.pkl")
# df_test = pd.read_pickle("test-title.pkl")

get_bigrams(df_train, tokenized, 'bigram col')
get_bigrams(df_test, tokenized, 'bigram col')

df_train[tokenized] = df_train[tokenized]+df_train['bigram col']
df_test[tokenized] = df_test[tokenized]+df_test['bigram col']

# df_train.to_pickle("train-title-bigram--2.pkl")
# df_train.to_pickle("test-title-bigram--2.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")
# df_train.to_pickle("train-title-bigram--2-predicted.pkl")
# df_test.to_pickle("test-title-bigram--2-predicted.pkl")

# df_train = pd.read_pickle("train-title-bigram--2-predicted.pkl")
# df_test = pd.read_pickle("test-title-bigram--2-predicted.pkl")

print("Accuracies using title column with bigrams")
metrics(df_train, df_test, class_col, bigram_predicted)

# title with bigrams and stopwords removal
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
title = 'title'
tokenize(df_train, title, tokenized)
tokenize(df_test, title, tokenized)
# df_train = pd.read_pickle("train-title.pkl")
# df_test = pd.read_pickle("test-title.pkl")

remove_stopwords(df_train, tokenized, tokenized)
remove_stopwords(df_test, tokenized, tokenized)

get_bigrams(df_train, tokenized, 'bigram col')
get_bigrams(df_test, tokenized, 'bigram col')

df_train[tokenized] = df_train[tokenized]+df_train['bigram col']
df_test[tokenized] = df_test[tokenized]+df_test['bigram col']

# df_train.to_pickle("train-title-bigram-sw.pkl")
# df_train.to_pickle("test-title-bigram-sw.pkl")

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col=bigram_predicted)
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col=bigram_predicted)
print("done")

# df_train.to_pickle("train-title-bigram-sw-predicted.pkl")
# df_test.to_pickle("test-title-bigram-sw-predicted.pkl")

# df_train = pd.read_pickle("train-title-bigram-sw-predicted.pkl")
# df_test = pd.read_pickle("test-title-bigram-sw-predicted.pkl")

print("Accuracies using title column with bigrams and stopword removal")
metrics(df_train, df_test, class_col, bigram_predicted)
'''
best so far
Accuracies using title column with bigrams
Accuracies using bigrams and unigrams without above preprocessing
'''

# train a model combining the best models. concatenating 

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
tokenize(df_train, content, tokenized)
tokenize(df_test, content, tokenized)

get_bigrams(df_train, tokenized, 'bigrams')
get_bigrams(df_test, tokenized, 'bigrams')

df_train[tokenized] += df_train['bigrams']
df_test[tokenized] += df_test['bigrams']

tokenize(df_train, 'title', 'tokenized title')
tokenize(df_test, 'title', 'tokenized title')

get_bigrams(df_train, 'tokenized title', 'bigrams')
get_bigrams(df_test, 'tokenized title', 'bigrams')

df_train[tokenized] += df_train['bigrams']
df_test[tokenized] += df_test['bigrams']

df_train[tokenized] += df_train['tokenized title']
df_test[tokenized] += df_test['tokenized title']


# df_train.to_pickle('train-combined.pkl')
# df_test.to_pickle('test-combined.pkl')

# df_train = pd.read_pickle('train-combined.pkl')
# df_test = pd.read_pickle('test-combined.pkl')

model = NaiveBayes()
print("training...")
model.fit(df_train, 1, class_col=class_col, text_col=tokenized)

print("getting predictions for training data")
model.predict(df_train, text_col=tokenized, predicted_col='combined predicted')
print("getting predictions for test data")
model.predict(df_test, text_col=tokenized, predicted_col='combined predicted')
print("done")
# print("saving databases")
# df_train.to_pickle("train-combined-predicted.pkl")
# df_test.to_pickle("test-combined-predicted.pkl")

# df_train = pd.read_pickle("train-combined-predicted.pkl")
# df_test = pd.read_pickle("test-combined-predicted.pkl")

print("Accuracies using title and content column with bigrams and unigrams")
metrics(df_train, df_test, class_col, 'combined predicted')

# 6b learn separate theta to title and content
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
tokenize(df_train, content, tokenized)
tokenize(df_test, content, tokenized)

get_bigrams(df_train, tokenized, 'bigrams')
get_bigrams(df_test, tokenized, 'bigrams')

df_train[tokenized] += df_train['bigrams']
df_test[tokenized] += df_test['bigrams']

tokenize(df_train, 'title', 'tokenized title')
tokenize(df_test, 'title', 'tokenized title')

get_bigrams(df_train, 'tokenized title', 'bigrams')
get_bigrams(df_test, 'tokenized title', 'bigrams')

df_train['tokenized title'] += df_train['bigrams']
df_test['tokenized title'] += df_test['bigrams']


# df_train.to_pickle('train-combined-b.pkl')
# df_test.to_pickle('test-combined-b.pkl')

# df_train = pd.read_pickle('train-combined-b.pkl')
# df_test = pd.read_pickle('test-combined-b.pkl')

model_text = NaiveBayes()
print("training... 1")
text_log_theta, log_priors = model_text.fit(df_train, 1, class_col=class_col, text_col=tokenized)
text_vocab = model_text.vocabulary
K = model_text.K


model_title = NaiveBayes()
print("training... 2")
title_log_theta, _ = model_title.fit(df_train, 1, class_col=class_col, text_col='tokenized title')
title_vocab = model_title.vocabulary

print("getting predictions")
predict_combined(df_train, K, tokenized, 'tokenized title', 'combined predicted', text_vocab, title_vocab, text_log_theta, title_log_theta, 
                 log_priors)
predict_combined(df_test, K, tokenized, 'tokenized title', 'combined predicted', text_vocab, title_vocab, text_log_theta, title_log_theta, 
                 log_priors)

# print("saving databases")
# df_train.to_pickle("train-combined-b-predicted.pkl")
# df_test.to_pickle("test-combined-b-predicted.pkl")

# df_train = pd.read_pickle("train-combined-b-predicted.pkl")
# df_test = pd.read_pickle("test-combined-b-predicted.pkl")

print("Accuracies using title and content column with bigrams and unigrams not with concatenation")
metrics(df_train, df_test, class_col, 'combined predicted')

#7 best vs base case
df_train['random'] = np.random.choice(14, size = m_train, replace=True)
df_test['random'] = np.random.choice(14, size = m_test, replace=True)
print("accuracy if we were predicting randomly")
metrics(df_train, df_test, class_col, 'random')

df_train['same'] = np.zeros(shape=(m_train,))
df_test['same'] = np.zeros(shape=(m_test,))
print("accuracy if we were predicting all the classes as 0")
metrics(df_train, df_test, class_col, 'same')

# 8 confusion matrix

cm_train = confusion_matrix(df_train[class_col], df_train['combined predicted'],)
disp1 = ConfusionMatrixDisplay(cm_train, display_labels=np.arange(14))
cm_test = confusion_matrix(df_test[class_col], df_test['combined predicted'])
disp2 = ConfusionMatrixDisplay(cm_test, display_labels=np.arange(14))

fig, axes = plt.subplots(1,2,figsize = (10, 5))
disp1.plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Confusion Matrix - train")
disp2.plot(ax=axes[1], cmap='Greens')
axes[1].set_title("Confusion Matrix - test")
plt.tight_layout()
plt.show()

# model recognises class 11(0-indexed) most accurately. it is probably easier to recognise the class 11 as compared to others

# Some feature engineering. Remove punctuation marks

print("Accuracies using title and content column with bigrams and unigrams with punctuations removed")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
tokenize(df_train, content, tokenized, keep_punctution=False)
tokenize(df_test, content, tokenized, keep_punctution=False)

get_bigrams(df_train, tokenized, 'bigrams')
get_bigrams(df_test, tokenized, 'bigrams')

df_train[tokenized] += df_train['bigrams']
df_test[tokenized] += df_test['bigrams']

tokenize(df_train, 'title', 'tokenized title', keep_punctution=False)
tokenize(df_test, 'title', 'tokenized title', keep_punctution=False)

get_bigrams(df_train, 'tokenized title', 'bigrams')
get_bigrams(df_test, 'tokenized title', 'bigrams')

df_train['tokenized title'] += df_train['bigrams']
df_test['tokenized title'] += df_test['bigrams']


# df_train.to_pickle('train-combined-b.pkl')
# df_test.to_pickle('test-combined-b.pkl')

# df_train = pd.read_pickle('train-combined-b.pkl')
# df_test = pd.read_pickle('test-combined-b.pkl')

model_text = NaiveBayes()
print("training... 1")
text_log_theta, log_priors = model_text.fit(df_train, 1, class_col=class_col, text_col=tokenized)
text_vocab = model_text.vocabulary
K = model_text.K


model_title = NaiveBayes()
print("training... 2")
title_log_theta, _ = model_title.fit(df_train, 1, class_col=class_col, text_col='tokenized title')
title_vocab = model_title.vocabulary

print("getting predictions")
predict_combined(df_train, K, tokenized, 'tokenized title', 'combined predicted', text_vocab, title_vocab, text_log_theta, title_log_theta, 
                 log_priors)
predict_combined(df_test, K, tokenized, 'tokenized title', 'combined predicted', text_vocab, title_vocab, text_log_theta, title_log_theta, 
                 log_priors)

# print("saving databases")
# df_train.to_pickle("train-combined-b-predicted.pkl")
# df_test.to_pickle("test-combined-b-predicted.pkl")

# df_train = pd.read_pickle("train-combined-b-predicted.pkl")
# df_test = pd.read_pickle("test-combined-b-predicted.pkl")

print("Accuracies using title and content column with bigrams and unigrams not with concatenation and punctuations removed")
metrics(df_train, df_test, class_col, 'combined predicted')

