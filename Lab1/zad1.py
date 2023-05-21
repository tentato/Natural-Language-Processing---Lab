with open("text.txt", "r", encoding="utf8") as f:
    text = f.readlines()
for line in text:
    print(line)

with open("tekst_nieparzyste.txt", "w", encoding="utf8") as f2:
    for index, line in enumerate(text):
        if ((index%2) == 1):
            f2.write(line)