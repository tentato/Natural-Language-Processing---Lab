import re
import json

with open("text.txt", "r", encoding="utf8") as f:
    text = f.read()

#1
sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
# print(sentences)

#2
clear_sentences = []
for sentence in sentences:
    sentence = re.sub(' +', ' ', sentence)
    sentence = clear_sentences.append(sentence)
# print(clear_sentences)

#3
word_count = len(re.findall(r'\w+', text))
# print(word_count)

#4
word_list = re.split(r"\s+", text)
# print(word_list)

#5
znaki_diakrytyczne = ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ż", "ź"]
clear_words = []
for word in word_list:
    r = re.compile(r"[^a-zA-Z0-9-ąćęłńóśżź]+")
    word = r.sub("", word)
    clear_words.append(word)
# print(clear_words)


#6
sentences_count = len(clear_sentences)
# print(sentences_count)

#7
result_name = "result.json"
dictionary = {
    "ilosc_zdan": sentences_count,
    "lista_oczyszczonych_zdan": clear_sentences,
    "ilosc_slow": word_count,
    "lista_oczyszczonych_slow": clear_words
}

json_object = json.dumps(dictionary, indent=4)
 
with open(result_name, "w") as outfile:
    outfile.write(json_object)