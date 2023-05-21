import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

# Zadanie odrabiane z powodu nieobecności. Zawiera modyfikacje wskazane przez prowadzącego.

#### ZADANIE 1 #######################################################################

tokenizer = nltk.tokenize.TreebankWordTokenizer()
s_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

text = pd.read_csv("imbd.csv",delimiter=",")

text.content = text.content.apply(str.lower)
raw_tokens = text.content.apply(tokenizer.tokenize)
clear_tokens = []
for tokens in raw_tokens:
    clear_token = []
    for token in tokens:
        r = re.compile(r"[^a-zA-Z0-9]+")
        token = r.sub("", token)
        clear_token.append(token)
    clear_tokens.append(clear_token)
text["tokens"] = clear_tokens
    
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

text["tokens"] = [" ".join(tokens) for tokens in text["tokens"]]
text["tokens_no_stopwords"] = [" ".join(tokens) for tokens in text["tokens_no_stopwords"]]
text["tokens_stemm"] = [" ".join(tokens) for tokens in text["tokens_stemm"]]
text["tokens_lem"] = [" ".join(tokens) for tokens in text["tokens_lem"]]

clear_tokens = []
clear_tokens_sw = []
clear_tokens_st = []
clear_tokens_le = []
for sentence in text["tokens"]:
    sentence = re.sub(' +', ' ', sentence)
    sentence = clear_tokens.append(sentence)
for sentence in text["tokens_no_stopwords"]:
    sentence = re.sub(' +', ' ', sentence)
    sentence = clear_tokens_sw.append(sentence)
for sentence in text["tokens_stemm"]:
    sentence = re.sub(' +', ' ', sentence)
    sentence = clear_tokens_st.append(sentence)
for sentence in text["tokens_lem"]:
    sentence = re.sub(' +', ' ', sentence)
    sentence = clear_tokens_le.append(sentence)

text["tokens"] = clear_tokens
text["tokens_no_stopwords"] =clear_tokens_sw
text["tokens_stemm"] = clear_tokens_st
text["tokens_lem"] = clear_tokens_le

text.to_csv("result.csv", index=False)
print(f"Results saved as result.csv")

#### ZADANIE 2 #######################################################################

dataset = pd.read_csv("result.csv",delimiter=",")
random_state = 10
sets = ["tokens","tokens_no_stopwords","tokens_stemm","tokens_lem"]
models = [MLPClassifier(), SVC()]

for set in sets:
    for model in models:
        X_train, X_test, y_train, y_test = train_test_split(dataset[set], dataset.label, test_size=0.3, random_state=random_state)

        cv = CountVectorizer()
        X = cv.fit_transform(X_train).toarray()
        y = y_train
        model.fit(X, y)

        y_pred = model.predict(cv.transform(X_test).toarray())
        score = accuracy_score(y_test, y_pred)
        print(f"Model: {model}, Dataset: {set}, Accuracy score: {score}\n")

# Obserwacje:        
#   Wyniki dla klasyfikatora SVC są lepsze niż dla MLP.
#   Zastosowanie stemmingu i lematyzacji ma pozytywny wpływ na wynik metryki accuraccy score.

##### ZADANIE 3

dataset = pd.read_csv("result.csv",delimiter=",")
random_state = 10
sets = ["tokens","tokens_no_stopwords","tokens_stemm","tokens_lem"]
model = MLPClassifier()

set_results = []

for set in sets:
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(dataset[set]).toarray()
    y = dataset.label
    
    kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)
    splits = kfold.split(X,y)
    
    folds_results = []
    for n,(train_index,test_index) in enumerate(splits):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(x_train_fold, y_train_fold)

        y_pred = model.predict(x_test_fold)
        # score = accuracy_score(y_test_fold, y_pred)
        score = precision_recall_fscore_support(y_test_fold, y_pred, average='weighted')
        # print(f"Score: {score}, n: {n}")
        folds_results.append(score)

    set_results.append(folds_results)

print(set_results)