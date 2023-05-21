import pandas as pd
import sklearn
import nltk
from nltk.corpus import stopwords
stopw = stopwords.words('english')
nltk.download('wordnet')

# zadanie 1 

tokenizer = nltk.tokenize.TreebankWordTokenizer()
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

imbd = pd.read_csv("imbd.csv")
imbd.content = imbd.content.apply(str.lower)
imbd["tokens"] = imbd.content.apply(tokenizer.tokenize)

def stem(sentence: list[str]):
    return [stemmer.stem(word) for word in sentence]

def lemmatize(sentence: list[str]):
    return [lemmatizer.lemmatize(word) for word in sentence]

def no_stop_words(sentence: list[str]):
    return [word for word in sentence if word not in stopw]
    
imbd["no_stopwords"] = imbd.tokens.apply(no_stop_words)
imbd["stemmed"] = imbd["no_stopwords"].apply(stem)
imbd["lemmatize"] = imbd["no_stopwords"].apply(lemmatize)
print(imbd)

imbd["no_stopwords"] = imbd.no_stopwords.apply(lambda x: " ".join(x))
imbd["stemmed"] = imbd["stemmed"].apply(lambda x: " ".join(x))
imbd["lemmatize"] = imbd["lemmatize"].apply(lambda x: " ".join(x))
imbd.to_csv("imbd_processed.csv", index=False)
# print(len(imbd.content.to_list()))




# zadanie 2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(imbd.index, imbd.label, test_size=0.3, random_state=99)


print(X_train, X_test, y_train, y_test)

def test_accuracy(name: str):
    cv = CountVectorizer()
    base = cv.fit_transform(imbd[name][X_train])
    base_2 = base.toarray()
    print(base_2.shape)

    model = clone(MultinomialNB())
    model.fit(base_2, y_train)

    y_pred = model.predict(cv.transform(imbd[name][X_test]).toarray())
    score = accuracy_score(y_test, y_pred)
    print(y_test)
    print(score)


for name in ("content", "no_stopwords", "stemmed", "lemmatize"):
    test_accuracy(name)


# Zadanie 3
def test_accuracy_2(name: str):
    skf = StratifiedKFold()

    for i, (train_index, test_index) in enumerate(skf.split(imbd.index, imbd.label)):
        clf = MLPClassifier()
        vec = TfidfVectorizer()
        base = vec.fit_transform(imbd[name][train_index])
        base_2 = base.toarray()

        clf.fit(base_2, imbd.label[train_index])

        y_pred = clf.predict(vec.transform(imbd[name][test_index]).toarray())
        score = accuracy_score(imbd.label[test_index], y_pred)

        print(score)

        print("end", name)
    ...

for name in ("content", "no_stopwords", "stemmed", "lemmatize"):
    test_accuracy_2(name)