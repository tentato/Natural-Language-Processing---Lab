import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim

# Zadanie 1

nltk.download('punkt')

directory_adam = 'poezja/adam/'
directory_jan = 'poezja/jan/'
directory_juliusz = 'poezja/juliusz/'
directory_poezja = 'poezja/'

filenames_adam =  os.listdir(directory_adam)
filenames_jan =  os.listdir(directory_jan)
filenames_juliusz =  os.listdir(directory_juliusz)

texts = []
texts_adam = []
texts_jan = []
texts_juliusz = []

for file in filenames_adam:
    with open(directory_adam+file, "r", encoding="utf8") as f:
        text = f.read()
        texts.append(text)
        texts_adam.append(text)
# DS Store file removed from folder
for file in filenames_jan:
    with open(directory_jan+file, "r", encoding="utf8") as f:
        text = f.read()
        texts.append(text)
        texts_jan.append(text)
for file in filenames_juliusz:
    with open(directory_juliusz+file, "r", encoding="utf8") as f:
        text = f.read()
        texts.append(text)
        texts_juliusz.append(text)

with open(directory_poezja+"stopwordsPL.txt", "r", encoding="utf8") as f:
    stopwords = f.readlines()
stopwords_clear = []
for word in stopwords:
    stopwords_clear.append(word.replace("\n", ""))
stopwords = set(stopwords_clear)

tokens = []
tokens_adam = []
tokens_jan = []
tokens_juliusz = []
for text in texts:
    alpha_token = []
    token_list = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    for i in token_list:
        alpha_token.append(''.join(e for e in i if e.isalnum()))
    tokens.append(alpha_token)

for text in texts_adam:
    token_list = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    alpha_token = []
    for i in token_list:
        alpha_token.append(''.join(e for e in i if e.isalnum()))
    tokens_adam.append(alpha_token)

for text in texts_jan:
    token_list = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    alpha_token = []
    for i in token_list:
        alpha_token.append(''.join(e for e in i if e.isalnum()))
    tokens_jan.append(alpha_token)

for text in texts_juliusz:
    token_list = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    alpha_token = []
    for i in token_list:
        alpha_token.append(''.join(e for e in i if e.isalnum()))
    tokens_juliusz.append(alpha_token)

model = gensim.models.Word2Vec(sentences=tokens, vector_size=32, window=5, min_count=1, workers=4)
model.train(tokens_adam, total_examples=model.corpus_count, epochs=4)
model.train(tokens_jan, total_examples=model.corpus_count, epochs=4)
model.train(tokens_juliusz, total_examples=model.corpus_count, epochs=4)

print(model.wv.similarity('wiatr', 'fale'))
print(model.wv.similarity('trawie', 'zioła'))
print(model.wv.similarity('zbroja', 'szalonych'))
print(model.wv.similarity('cichym', 'szeptem'))

# 0.9529961
# 0.35018194
# 0.39924064
# 0.8062433

# Pierwsze i ostatnie pary są najbardziej podobne.

# Zadanie 2

