import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.arlstem import ARLSTem
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import PrecisionRecallDisplay

#### ZADANIE 1 #######################################################################

tokenizer = nltk.tokenize.TreebankWordTokenizer()
s_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
# stemmer = ARLSTem()
lemmatizer = WordNetLemmatizer()

text = pd.read_csv("imbd.csv",delimiter=",")

text.content = text.content.apply(str.lower)
text["tokens"] = text.content.apply(tokenizer.tokenize)
tokens_no_stopwords = []
for sentence in text["tokens"]:
    tokens_no_stopwords.append([word for word in sentence if word not in s_words])
text["tokens_no_stopwords"] = tokens_no_stopwords

tokens_stemm = []
for sentence in text["tokens_no_stopwords"]:
    tokens_stemm.append([stemmer.stem(word) for word in sentence])
text["tokens_stemm"] = tokens_stemm
tokens_lem = []

for sentence in text["tokens_no_stopwords"]:
    tokens_lem.append([stemmer.stem(word) for word in sentence])
text["tokens_lem"] = tokens_lem

text.to_csv("result.csv", index=False)
print(f"Results saved as result.csv")

#### ZADANIE 2 #######################################################################

dataset = pd.read_csv("result.csv",delimiter=",")
random_state = 10


sets = ["tokens","tokens_no_stopwords","tokens_stemm","tokens_lem"]
models = [MLPClassifier(), SVC()]

for set in sets:
    for model in models:
        print(f"Model: {model}")
        X_train, X_test, y_train, y_test = train_test_split(dataset[set], dataset.label, test_size=0.3, random_state=random_state)

        cv = CountVectorizer()
        X = cv.fit_transform(X_train).toarray()
        y = y_train
        model.fit(X, y)

        y_pred = model.predict(cv.transform(X_test).toarray())
        score = accuracy_score(y_test, y_pred)
        print(f"Score: {score}")

##### ZADANIE 3

for set in sets:
    for model in models:
        print(f"Model: {model}")
        
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(dataset[set]).toarray()
        y = dataset.label
        
        kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,random_state=random_state)
        splits = kfold.split(X,y)

        for n,(train_index,test_index) in enumerate(splits):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            model.fit(x_train_fold, y_train_fold)

            y_pred = model.predict(x_test_fold)
            score = accuracy_score(y_test_fold, y_pred)
            print(f"Score: {score}")
            # display = PrecisionRecallDisplay.from_estimator(
            #     classifier, X_test, y_test, name="LinearSVC"
            # )
            # _ = display.ax_.set_title("2-class Precision-Recall curve")