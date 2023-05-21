import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
# from gensim.models import Word2Vec

nltk.download('punkt')

directory_adam = 'lab3/poezja/adam/'
directory_jan = 'lab3/poezja/jan/'
directory_juliusz = 'lab3/poezja/juliusz/'
directory_poezja = 'lab3/poezja/'

filenames_adam =  os.listdir(directory_adam)
filenames_jan =  os.listdir(directory_jan)
filenames_juliusz =  os.listdir(directory_juliusz)

texts = []
texts_adam = []
texts_jan = []
texts_juliusz = []

for file in filenames_adam:
    print(file)
    with open(directory_adam+file, "r", encoding="utf8") as f:
        texts.append(f.read())
        texts_adam.append(f.read())
# DS Store file removed from folder
for file in filenames_jan:
    print(file)
    with open(directory_jan+file, "r", encoding="utf8") as f:
        texts.append(f.read())
        texts_jan.append(f.read())
for file in filenames_juliusz:
    print(file)
    with open(directory_juliusz+file, "r", encoding="utf8") as f:
        texts.append(f.read())
        texts_juliusz.append(f.read())

# print(texts[0])
# exit()

with open(directory_poezja+"stopwordsPL.txt", "r", encoding="utf8") as f:
    stopwords = f.readlines()
stopwords_clear = []
for word in stopwords:
    stopwords_clear.append(word.replace("\n", ""))
stopwords = set(stopwords_clear)

# 1 token = 1 text
tokens = []
tokens_adam = []
tokens_jan = []
tokens_juliusz = []
for text in texts:
    token_list = [i for i in word_tokenize(text.lower()) if i not in stopwords]
    alpha_token = []
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

# print(tokens[0])
# print(len(tokens))

model = gensim.models.Word2Vec(sentences=tokens, vector_size=16, window=5, min_count=1)
# model.train(tokens_adam, total_examples=model.corpus_count, epochs=8)
model.train(tokens_jan, total_examples=model.corpus_count, epochs=16)
# model.train(tokens_juliusz, total_examples=model.corpus_count, epochs=8)

print(model.wv.similarity('wiatr', 'fale'))
print(model.wv.similarity('trawie', 'zioła'))
print(model.wv.similarity('zbroja', 'szalonych'))
print(model.wv.similarity('cichym', 'szeptem'))

# wyniki podobieństwa bardzo niskie, pierwsze i ostatnie są najbardziej podobne
# 3.11
# 3.10