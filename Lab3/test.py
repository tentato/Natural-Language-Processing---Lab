import nltk
import numpy as np
from numpy import dot
from numpy.linalg import norm

from nltk.tokenize import word_tokenize
import glob

from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


nltk.download('punkt')

stopwords = []
with open("poezja/stopwordsPL.txt", "r") as file:
    stopwords = file.read().splitlines()

# print("Stopwords")
# print(stopwords)

files_adam = glob.glob("poezja/adam/*.txt")
files_jan = glob.glob("poezja/jan/*.txt")
files_juliusz = glob.glob("poezja/juliusz/*.txt")

all_files_paths = [*files_adam, *files_jan, *files_juliusz]

def read_file(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        return file.read()

def filter_alnum(token):
    return ''.join(c for c in token if c.isalnum())


# print(files_adam)


adam_tokens_per_file = []
for file_path in files_adam:
    text = read_file(file_path)
    tokens = word_tokenize(text)
    tokens_sanitized = [token for token in tokens if token not in stopwords]
    tokens_sanitized_only_alnum = [filter_alnum(token) for token in tokens_sanitized]
    adam_tokens_per_file.append( tokens_sanitized_only_alnum)

jan_tokens_per_file = []
for file_path in files_jan:
    text = read_file(file_path)
    tokens = word_tokenize(text)
    tokens_sanitized = [token for token in tokens if token not in stopwords]
    tokens_sanitized_only_alnum = [filter_alnum(token) for token in tokens_sanitized]
    jan_tokens_per_file.append( tokens_sanitized_only_alnum)

juliusz_tokens_per_file = []
for file_path in files_juliusz:
    text = read_file(file_path)
    tokens = word_tokenize(text)
    tokens_sanitized = [token for token in tokens if token not in stopwords]
    tokens_sanitized_only_alnum = [filter_alnum(token) for token in tokens_sanitized]
    juliusz_tokens_per_file.append( tokens_sanitized_only_alnum)

all_sentences = [*adam_tokens_per_file, *jan_tokens_per_file, *juliusz_tokens_per_file]


model = Word2Vec(sentences=all_sentences, vector_size=32, window=5, min_count=1, workers=4)

# # model.save("word2vec.model.adam")
# # model = Word2Vec.load("word2vec.model.adam")
# model.train(adam_tokens_per_file, total_examples=model.corpus_count, epochs=4)
# model.train(jan_tokens_per_file, total_examples=model.corpus_count, epochs=4)
# model.train(juliusz_tokens_per_file, total_examples=model.corpus_count, epochs=4)

# # similarity
# print(model.wv.similarity('wiatr','fale'))          # 0.97967446
# print(model.wv.similarity('trawie','zioła'))        # 0.9360799
# print(model.wv.similarity('zbroja','szalonych'))    # 0.9207863
# print(model.wv.similarity('cichym','szeptem'))      # 0.91029173

# * zadanie 2
all_vectors = []
for doc_path in all_files_paths:
    text = read_file(doc_path)
    tokens = word_tokenize(text)
    tokens_sanitized = [token for token in tokens if token not in stopwords]
    tokens_sanitized_only_alnum = [filter_alnum(token) for token in tokens_sanitized]
    vectors = []
    for word in tokens_sanitized_only_alnum:
        vector = model.wv[word]
        vectors.append(vector)
    mean_vectors = np.mean(vectors, axis=0)
    all_vectors.append(mean_vectors)
    print(doc_path)
    print(mean_vectors)

cos_sim_all = []
for idx_a, a in enumerate(all_vectors):
    for idx_b, b in enumerate(all_vectors):
        if idx_a == idx_b:
            continue
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        output = {
            "file_1": all_files_paths[idx_a],
            "file_2": all_files_paths[idx_b],
            "cos_sim": cos_sim
        }
        cos_sim_all += [output]

mini = 100
mini_file_1_name = ''
mini_file_2_name = ''
maxi = 0
maxi_file_1_name = ''
maxi_file_2_name = ''

for sim_single in cos_sim_all:
    file_1_name = sim_single['file_1']
    file_2_name = sim_single['file_2']
    cos_sim = sim_single['cos_sim']

    if mini > cos_sim:
        mini = cos_sim
        mini_file_1_name = file_1_name
        mini_file_2_name = file_2_name

    if maxi < cos_sim:
        maxi = cos_sim
        maxi_file_1_name = file_1_name
        maxi_file_2_name = file_2_name

# print(cos_sim_all)
print("MINI")
print(mini)
print(mini_file_1_name)
print(maxi_file_2_name)

print("MAXI")
print(maxi)
print(maxi_file_1_name)
print(maxi_file_2_name)


# * zadanie 3

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_sentences)]
# model = Doc2Vec(documents, vector_size=32, window=5, min_count=1, workers=4)
# model.train(documents, total_examples=model.corpus_count, epochs=4)

# all_vectors = []
# for doc_path in all_files_paths:
#     text = read_file(doc_path)
#     tokens = word_tokenize(text)
#     tokens_sanitized = [token for token in tokens if token not in stopwords]
#     tokens_sanitized_only_alnum = [filter_alnum(token) for token in tokens_sanitized]
#     vectors = []
#     for word in tokens_sanitized_only_alnum:
#         vector = model.wv[word]
#         vectors.append(vector)
#     mean_vectors = np.mean(vectors, axis=0)
#     all_vectors.append(mean_vectors)
#     print(doc_path)
#     print(mean_vectors)

# cos_sim_all = []
# for idx_a, a in enumerate(all_vectors):
#     for idx_b, b in enumerate(all_vectors):
#         if idx_a == idx_b:
#             continue
#         cos_sim = dot(a, b)/(norm(a)*norm(b))
#         output = {
#             "file_1": all_files_paths[idx_a],
#             "file_2": all_files_paths[idx_b],
#             "cos_sim": cos_sim
#         }
#         cos_sim_all += [output]

# mini = 100
# mini_file_1_name = ''
# mini_file_2_name = ''
# maxi = 0
# maxi_file_1_name = ''
# maxi_file_2_name = ''

# for sim_single in cos_sim_all:
#     file_1_name = sim_single['file_1']
#     file_2_name = sim_single['file_2']
#     cos_sim = sim_single['cos_sim']

#     if mini > cos_sim:
#         mini = cos_sim
#         mini_file_1_name = file_1_name
#         mini_file_2_name = file_2_name

#     if maxi < cos_sim:
#         maxi = cos_sim
#         maxi_file_1_name = file_1_name
#         maxi_file_2_name = file_2_name

# # print(cos_sim_all)
# print("MINI")
# print(mini)
# print(mini_file_1_name)
# print(maxi_file_2_name)

# print("MAXI")
# print(maxi)
# print(maxi_file_1_name)
# print(maxi_file_2_name)
