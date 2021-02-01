import numpy as np
import nltk
import re
import os
import os.path as op
from nltk.stem.snowball import SnowballStemmer
#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2
#test kodudur.
#test edilmek istenen HAM VERİ(.txt) içeren klasör veya direkt .txt dosyası pathi kodda verilmelidir. 
#3. sırada çalıştırılması gereken koddur.

from collections import Counter
from scipy.sparse import csr_matrix

import pickle


def load_pkl(path: str):
    with open(path, 'rb') as f:
        lr_model = pickle.load(f)
        count_vectorizer = pickle.load(f)
    return lr_model, count_vectorizer


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        """
        LogisticRegression constructor fonksiyonu ve ilk değerlerin atanması.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Gelen sample ve class bilgilerine göre ağırlıkların güncellenmesini sağlamaktadır.
        """
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
        """
        Parametre ile gönderilen yeni bir sample'ın önceden eğitilmiş bir lojistik regression modeli
        tarafından sınıfının tahmin edilmesini sağlar.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        """
        Sigmoid fonksiyonu gönderilen sayısal değerleri (0-1) aralığında bir değere eşitlemek için kullanılır.
        """
        return 1 / (1 + np.exp(-x))


class CountVectorizer:
    def __init__(self):
        """
        Constructor metodu ile ilk değerlerin atanması.
        """
        self.vocabulary = dict()

    def fit(self, ls_texts: list):
        """
        Gönderilen metinlerden sözcüklerin alınarak vectorizer'e ait yeni bir sözcük haznesinin oluşturulması.
        """
        self.unique_words = set()

        for sentence in ls_texts:
            for word in sentence.split(' '):
                if len(word) >= 2:
                    self.unique_words.add(word)

        for index, word in enumerate(sorted(list(self.unique_words))):
            self.vocabulary[word] = index

        return self.vocabulary

    def transform(self, ls_texts: list):
        """
        Önceden oluşturulmuş sözcük haznesine göre, gelen yeni textlerin vektörlere dönüştürülmesini sağlar.
        """
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
        """
        Gönderilen text listesi üzerinden kelime haznesinin oluşturulmasını ve oluşturulan kelime haznesine göre
        metinlerin vektörlerinin çıkarılmasını sağlamaktadır.
        """
        self.fit(ls_texts)
        return self.transform(ls_texts)


def get_files_path_list_on_dir(*paths) -> list:
    ls = []
    for path in paths:
        for path_file in os.listdir(path):
            ls.append(op.join(path, path_file))
    return ls


def get_txt_data(path: str) -> str:
    with open(path, 'r') as f:
        data = f.read()
    return data


def preprocess_text(txt: str) -> str:
    """
    Amaç: txt adlı metinden gereksiz karakterlerin ve fazla olan boşlukların silinmesini sağlamaktadır.

    txt_new = re.sub("[^a-zA-Z ]", ' ', txt):
        Alfabede yer almayan (İngiliz alfabesi) ve boşluk olmayan tüm karakterleri metinden siler.

    txt_new = re.sub(' [abcdefghjklmnopqrstuvwxyz] ', ' ', txt_new):
        Tek karakterleri metinden siler.

    txt_new = re.sub(' ( )+', ' ', txt_new):
        Fazla boşlukların atılmasını sağlamaktadır.
    """
    txt_new = re.sub("[^a-zA-Z ]", ' ', txt)
    txt_new = re.sub(' [abcdefghjklmnopqrstuvwxyz] ', ' ', txt_new)  # Except i character, abcdefghijklmnopqrstuvwxyz
    txt_new = re.sub(' ( )+', ' ', txt_new)
    return txt_new


def clean_words(txt: str, ls_stopwords: tuple) -> str:
    """
    Amaç: txt adlı metinden, ls_stopwords adlı listede (tuple) olan kelimelerin temizlenmesidir.
        Örnek:
            txt: "Dear, Marry. Have you ever seen our village?", ls_stopwords: ("you", "ever", "our")
            Çıktı: "Dear, Marry. Have seen village?"

    Stopwords:
        Metinlerde sıklıkla kullanılan ancak metnin anlamına çok az anlam katan sözcükleri tutan listedir.
        Örneğin “the”, “is”, “to”, “at” gibi ifadelerin cümlenin anlamına etkisi yok denecek kadar azdır,
        dolayısıyla metinden bu kelimelerin çıkarılması tercih edilir.
    """
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


def predict_sample(lr, cv, txt, stopwords):
    text_active = txt.lower()
    text_active = preprocess_text(text_active)
    text_active = clean_words(text_active, stopwords)
    # text_active = get_stemming_text(text_active)

    X_test = cv.transform([text_active]).toarray()
    Y_pred = lr.predict(X_test)[0]
    return "Spam" if Y_pred == 1 else "Ham"


def predict_samples(lr, cv, stopwords, path_scan):
    dc = dict()
    for root, dirs, files in os.walk(path_scan):
        if not files:
            continue

        for file in files:
            if not file.endswith('.txt'):
                continue

            path = op.join(root, file)
            txt = get_txt_data(path)
            dc[path] = predict_sample(lr, cv, txt, stopwords)
    return dc


if __name__ == "__main__":
    
    #02 numaralı kodda kaydettiğimiz modeli yüklüyoruz.
    path_pkl = "res/classifier_countvectorizer_2.pkl"
    
    #test etmek istediğimiz txt formatında veri içeren klasör pathi verilmelidir.
    #aynı şekilde test edilmesi istenen txt verisinin pathi de verilebilir. 
    #kod hem klasör hem de .txt için işlem yapabilmektedir.
    path_scan = r"res/unseen2_tamamen_ham_test_data"

    lr, cv = load_pkl(path_pkl)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    stopwords = nltk.corpus.stopwords.words('english')

    if op.isfile(path_scan):
        txt = get_txt_data(path_scan)
        print(predict_sample(lr, cv, txt, stopwords))
    else:
        print(predict_samples(lr, cv, stopwords, path_scan))
        
#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2