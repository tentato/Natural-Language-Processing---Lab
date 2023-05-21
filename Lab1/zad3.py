import json
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
 
with open('result.json', 'r') as openfile:
    json_object = json.load(openfile)
 
clear_words = json_object['lista_oczyszczonych_slow']
clear_words_joined = " ".join(clear_words)

#1
mask1 = np.array(Image.open(path.join("maska.jpg")))
wc = WordCloud(background_color="white", repeat=True, mask=mask1)

# generate word cloud
wc.generate(clear_words_joined)

# store to file
wc.to_file("chmura.png")




#2
with open("stopwordsPL.txt", "r", encoding="utf8") as f:
    stopwords = f.readlines()
stopwords_clear = []
for word in stopwords:
    stopwords_clear.append(word.replace("\n", ""))
stopwords = set(stopwords_clear)

wc = WordCloud(background_color="white", stopwords=stopwords, repeat=True, mask=mask1)

# generate word cloud
wc.generate(clear_words_joined)

# store to file
wc.to_file("lepsza_chmura.png")

# Na apierwszy rzut oka widać, że w lepszej chmurze widoczne są "ciekawsze" słowa. 
# W pierwszej chmurze najwięcej jest przyimków, łączników i krótkich słów.
# Z drugiej chmury można wywnioskować czego dotyczy tekst.