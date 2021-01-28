#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2
#Bu kodda LogisticRegression ve CountVectorizer icin hazir kutuphaneler kullanilmamaktadir.
#Bu sebeple kodun calismasi 1.30 dakika civari surmektedir.

#importing libraries
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from collections import Counter
from scipy.sparse import csr_matrix

#reading the npz format data in the code
def get_npz_data(path_npz: str) -> (np.ndarray, np.ndarray, int):
    data = np.load(path_npz)
    texts = data["texts"]
    classes = data["classes"].astype(np.int8)
    len_rows = classes.shape[0]
    return texts, classes, len_rows

#preprocessing the texts in txt files
def preprocess_text(txt: str) -> str:
    txt_new = re.sub("[^a-zA-Z ]", ' ', txt)
    txt_new = re.sub(' [abcdefghjklmnopqrstuvwxyz] ', ' ', txt_new)  # Except i character, abcdefghijklmnopqrstuvwxyz
    txt_new = re.sub(' ( )+', ' ', txt_new)
    return txt_new


def clean_words(txt: str, ls_stopwords: tuple) -> str:
    ls_words = txt.split(' ')
    ls_deleted_index = []
    for i in range(len(ls_words)):
        if ls_words[i] in ls_stopwords:
            ls_deleted_index.append(i)
    i = 0
    for index in ls_deleted_index:
        del ls_words[index+i]
        i -= 1
    return ' '.join(ls_words)


def get_stemming_text(txt: str) -> str:
    stemmer = SnowballStemmer("english")
    ls_words = txt.split(' ')
    for i in range(len(ls_words)):
        ls_words[i] = stemmer.stem(ls_words[i])
    return ' '.join(ls_words)

#Implementation of the LogisticRegression without using sk-learn
class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate output variable (y) with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) #derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  #derivative w.r.t bias
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#Implementation of the CountVectorizer without using sk-learn CountVectorizer library
class CountVectorizer:
    def __init__(self):
        self.vocabulary = dict()

    def fit(self, ls_texts: list):
        self.unique_words = set()

        for sentence in ls_texts:
            for word in sentence.split(' '):
                if len(word) >= 2:
                    self.unique_words.add(word)

        for index, word in enumerate(sorted(list(self.unique_words))):
            self.vocabulary[word] = index

        return self.vocabulary

    def transform(self, ls_texts: list):
        row, col, val = [], [], []

        for idx, sentence in enumerate(ls_texts):
            count_word = dict(Counter(sentence.split(' ')))

            for word, count in count_word.items():
                if len(word) >= 2:
                    col_index = self.vocabulary.get(word)
                    if col_index is not None:
                        row.append(idx)
                        col.append(col_index)
                        val.append(count)

        return csr_matrix((val, (row, col)), shape=(len(ls_texts), len(self.vocabulary)))

    def fit_transform(self, ls_texts: list):
        self.fit(ls_texts)
        return self.transform(ls_texts)


if __name__ == "__main__":
    
    #giving the dataset paths
    path_npz_train = "res/training.npz"
    
    #with another test dataset, the txt files must be entering the txts_to_npz.py code first
    #and the output will be the npz format and the npz file path must given in this part.
    path_npz_test = "res/development.npz"

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

    texts, classes, len_rows = get_npz_data(path_npz_train)

    ls_texts = []
    for i in range(len_rows):
        text_active = texts[i].lower()
        text_active = preprocess_text(text_active)
        text_active = clean_words(text_active, stopwords)
        # text_active = get_stemming_text(text_active)
        ls_words_active = text_active.split(' ')
        ls_texts.append(ls_words_active)

    words_whole = [' '.join(words) for words in ls_texts]
    cv = CountVectorizer()
    X = cv.fit_transform(words_whole).toarray()
    print(X)

    lr = LogisticRegression()
    lr.fit(X, classes)

    texts_test, classes_test, len_rows_test = get_npz_data(path_npz_test)

    ls_texts = []
    for i in range(len_rows_test):
        text_active = texts_test[i].lower()
        text_active = preprocess_text(text_active)
        text_active = clean_words(text_active, stopwords)
        # text_active = get_stemming_text(text_active)
        ls_words_active = text_active.split(' ')
        ls_texts.append(ls_words_active)

    words_whole = [' '.join(words) for words in ls_texts]
    X_test = cv.transform(words_whole).toarray()
    print(X_test)

    Y_pred = lr.predict(X_test)
    cm = confusion_matrix(classes_test, Y_pred)
    print(cm)

    sns.set()
    ax = sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    Accuracy = accuracy_score(classes_test, Y_pred)
    print('Accuracy in Percentage is =', Accuracy * 100)
    
    precision = precision_score(classes_test, Y_pred)
    print('Precision in Percentage is =', precision * 100)
    
    recall = recall_score(classes_test, Y_pred)
    print('Recall in Percentage is =', recall * 100)
    
    f1 = f1_score(classes_test, Y_pred)
    print('F1 Score in Percentage is =', f1 * 100)

#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2
