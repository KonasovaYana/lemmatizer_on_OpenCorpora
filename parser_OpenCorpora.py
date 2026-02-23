import json

def parse_dict_file(filename="dict.opcorpora.txt"):

    wordforms = {}  
    lemmas = {}    
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    total_sections = 0
    
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.isdigit():
            total_sections += 1
            i += 1

            first_line = lines[i].strip()
            if first_line:
                parts = first_line.split('\t')
                if len(parts) >= 2:
                    lemma = (parts[0].strip()).replace("Ё", "Е").replace("ё", "е")
                    pos_info = parts[1].strip()

                    pos = pos_info.split(',')[0] if ',' in pos_info else pos_info
                    pos = pos.split(' ')[0] if ' ' in pos else pos

                    if lemma not in lemmas:
                        lemmas[lemma] = [pos]
                    elif pos not in lemmas[lemma]: 
                        lemmas[lemma].append(pos)

                    wordforms[lemma.lower()] = (lemma, pos)

                    i += 1
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        word_line = (lines[i].strip()).replace("Ё", "Е").replace("ё", "е")
                        if word_line:
                            word_parts = word_line.split('\t')
                            if len(word_parts) >= 2:
                                wordform = word_parts[0].strip()
                                wordforms[wordform.lower()] = (lemma, pos)
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return wordforms, lemmas

wordforms, lemmas = parse_dict_file("dict.opcorpora.txt")
odict_data = {
    "lemmas": lemmas,
    "wordforms": {word: {"lemma": lemma, "pos": pos} for word, (lemma, pos) in wordforms.items()}
}
with open("dictionary", "w", encoding="utf-8") as f:
    json.dump(odict_data, f, ensure_ascii=False, indent=2)
