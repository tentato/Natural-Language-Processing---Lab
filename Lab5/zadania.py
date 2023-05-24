import requests
import json
import numpy as np
import scrapy
from scrapy.item import Item, Field
from bs4 import BeautifulSoup

# zadanie 1

response = requests.get('https://wolnelektury.pl/api/books/')
response_json = response.json()

random_books = []

random_books_indexes = np.random.randint(0, len(response_json), size=(20))
for i in random_books_indexes:
    random_books.append(response_json[i])
# print(random_books_indexes)
# print(random_books)
# Wypisz tytuł, treść, autora, gatunek oraz epokę dla każdego utworu.
for book in random_books:
    for attribute, value in book.items():
        if attribute == 'title' or attribute == 'author' or attribute == 'epoch':
            print(f"{attribute}: {value}")
        elif attribute == 'href':
            try:
                res = requests.get(value).json()
                content_link = res["txt"]
                content = requests.get(content_link)
                print(f"Treść: {content.text}")
            except:
                print("Treść niedostępna!")

# Spotify posiada bardzo fajne API. 
# Można pobawić się w przypisywanie tekstów piosenek do pory roku/samopoczucia.

# zadanie 2

print("ZADANIE 2")

start_url = "https://gazetawroclawska.pl/wiadomosci"

res = requests.get(start_url)
s = BeautifulSoup(res.content, 'html.parser')
title = str(s.find("h2").find("b")).replace("<b>","").replace("</b>", "")
print(title)
# "Plik robots.txt przekazuje robotom wyszukiwarek informacje, do których adresów URL w Twoje witrynie roboty te mogą uzyskać dostęp. 
# Używa się go głównie po to, aby witryna nie była przeciążona żądaniami." - https://developers.google.com/search/docs/crawling-indexing/robots/intro?hl=pl

# zadanie 3

# Opóźnienie jest po to, aby nie dostać bana 

# content = s
# all_divs = content.find_all("div")