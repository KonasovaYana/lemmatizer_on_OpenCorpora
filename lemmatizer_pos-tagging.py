import re
import json
import conllu 

OPEN_CORPORA_TO_UD = {
    'NOUN': ['NOUN', 'PROPN'], 'ADJF': ['ADJ', 'DET'], 'ADJS': 'ADJ', 'COMP': 'ADJ',
    'VERB': ['VERB', 'AUX'], 'INFN': 'VERB', 'PRTF': 'ADJ', 'PRTS': 'ADJ',
    'GRND': 'VERB', 'NUMR': 'NUM', 'ADVB': 'ADV', 'NPRO': ['PRON', 'DET'],
    'PRED': ['ADV', 'ADJ'], 'PREP': 'ADP', 'CONJ': ['CCONJ', 'SCONJ'],
    'PRCL': ['PART', 'AUX'], 'INTJ': 'INTJ', 'unknown': 'X'
}

RULES_INFINITIVE = [
    (r'ла$', 'ть'), (r'ла$', 'ти'), (r'ла$', 'чь'), (r'ла$', 'зти'),     
    (r'ся$', 'ться'), (r'ся$', 'тись'), (r'ся$', 'чься'), (r'лся$', 'ться'),   
    (r'лся$', 'тись'), (r'лся$', 'чься'), (r'л$', 'ть'), (r'л$', 'ти'), (r'л$', 'сти'), (r'л$', 'чь'),       
    (r'л$', 'зти'), (r'ли$', 'ть'), (r'ли$', 'ти'), (r'ли$', 'чь'), (r'ит$', 'ить'),     
    (r'ет$', 'еть'), (r'ет$', 'ети'), (r'ют$', 'ить'), (r'ят$', 'ить'), (r'ат$', 'ать'),     
    (r'у$', 'ть'), (r'ю$', 'ть'), (r'ешь$', 'ть'), (r'ешь$', 'ти'), (r'ишь$', 'ить'),       
    (r'лась$', 'ться'), (r'лась$', 'тись'), (r'лась$', 'чься'), ('есть', 'быть')
]

ENDINGS = [
    'ая', 'яя', 'ое', 'ее', 'ие', 'ые',
    'ого', 'его', 'ому', 'ему', 'ым', 'им',
    'ом', 'ем', 'ую', 'юю', 'а', 'я', 'ый', 'ий', 'ой',
    'ть', 'ти', 'чь', 'ет', 'ют', 'ит', 'ат', 'ят',
    'ем', 'им', 'ете', 'ите', 'ут', 'ют', 'ал', 'ял',
    'ла', 'ло', 'ли', 'л', 'в', 'вши', 'вшись',
    'а', 'я', 'у', 'ю', 'ом', 'ем', 'е', 'и',
    'ы', 'ей', 'ам', 'ям', 'ами', 'ями', 'ах', 'ях',
    'ь', 'ия', 'ие', 'ий', 'ию', 'ием',
    'ой', 'ый', 'ий', 'ое', 'ее', 'ие', 'ые',
    'ыми', 'ими', 'ых', 'их'        
]

VERB_TAGS = ['VERB', 'GRND', 'PRTF', 'PRTS']

def add_ending_to_stem(stem, pos):
    stem = stem.lower().strip()

    if pos == "NOUN" or pos == "NUMR" or pos == "NPRO" or pos == "PREP" or pos == "CONJ" or pos == "PRCL" or pos == "INTJ": 
        return stem

    elif pos == "ADJF" or pos == "ADJS" or pos == "COMP" or pos == "PRTF" or pos == "PRTS": 
        return stem + 'ый'

    elif pos == "VERB" or pos == "INFN" or pos == "GRND":
        return stem + 'ть'
    
    elif pos == "ADVB" or pos == "PRED":
        return stem + 'о'
    

    
def levenshtein_distance(s1, s2):

    s1 = s1.lower()
    s2 = s2.lower()

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
        
            dp[i][j] = min(
                dp[i-1][j] + 1,      
                dp[i][j-1] + 1,   
                dp[i-1][j-1] + cost  
            )
    
    return dp[m][n]

def build_index(dictionary):
    index = {}
    for word, info in dictionary["wordforms"].items():
        word_lower = word.lower()
        if len(word_lower) >= 1:
            first_letter = word_lower[0]
            if first_letter not in index:
                index[first_letter] = []
            index[first_letter].append((word, info["pos"], word_lower))

    for letter in index:
        index[letter].sort(key=lambda x: len(x[2]))
    
    return index

def find_closest_word_indexed(target_word, dictionary, index):
    target_word = target_word.lower()
    target_len = len(target_word)
    first_letter = target_word[0] if target_word else ''

    candidates = []
    if first_letter in index:
        for word, pos, word_lower in index[first_letter]:
            if abs(len(word_lower) - target_len) <= 3:
                candidates.append((word, pos, word_lower))
    if len(candidates) < 20:
        for letter, items in index.items():
            if letter != first_letter:
                for word, pos, word_lower in items[:10]: 
                    if abs(len(word_lower) - target_len) <= 2:
                        candidates.append((word, pos, word_lower))
                        if len(candidates) >= 50:
                            break
            if len(candidates) >= 50:
                break
    
    min_distance = float('inf')
    closest_word = None
    pos = None
    
    for word, word_pos, word_lower in candidates:
        distance = levenshtein_distance(target_word, word_lower)
        if distance < min_distance:
            min_distance = distance
            closest_word = word
            pos = word_pos
            if distance == 0:
                break
    
    return closest_word, pos

def stem_word(word, endings = ENDINGS):
    
    word = word.lower().strip()
    if len(word) <= 3:
        return word

    for ending in endings:
        if word.endswith(ending):
            return word[:-len(ending)]
    
    return word

def word_in_dict(word, dictionary):
    word_lower = word.lower()
    word_upper = word.upper()

    if word_upper in dictionary.get("lemmas", {}):
        return True
    
    if word_lower in dictionary.get("wordforms", {}):
        return True
    
    return False

def normalize_verb_lemma(lemma, pos, dictionary, rules):
    is_verb = False
    if isinstance(pos, list):
        is_verb = any(tag in VERB_TAGS for tag in pos)
    else:
        is_verb = pos in VERB_TAGS
    if is_verb:
        for pattern, replacement in rules:
            if re.search(pattern, lemma.lower()):
                new_lemma = re.sub(pattern, replacement, lemma.lower())
                if word_in_dict(new_lemma, dictionary):
                    return new_lemma
    return lemma

def load_syntagrus_data(filepath):
    sentences_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for tokenlist in conllu.parse_incr(f):
            sentence_tokens = []
            true_tags_ud = []
            true_lemmas = []  
            for token in tokenlist:
                if token['upos'] != 'PUNCT':
                    sentence_tokens.append(token['form'])
                    true_tags_ud.append(token['upos'])
                    true_lemmas.append(token['lemma']) 
            if sentence_tokens:
                sentences_data.append((sentence_tokens, true_tags_ud, true_lemmas))
    return sentences_data

def evaluate_on_syntagrus(sentences_data, dictionary, mapping, index):
    total_tokens = 0
    correct_pos = 0
    correct_lemma = 0
    correct_both = 0
    lemma_errors = []
    pos_errors = [] 
    pos_error_by_tag = {}
    j=1
    for tokens, true_tags_ud, true_lemmas in sentences_data:
        total_tokens += len(tokens)
        for i, word in enumerate(tokens):
            pred_lemma, pred_pos_list = search_in_dict(word, dictionary)

            if pred_lemma is None:
                pred_lemma_norm = word.lower().replace('ё', 'е')
            else:
                pred_lemma = normalize_verb_lemma(pred_lemma, pred_pos_list, dictionary, RULES_INFINITIVE)
                pred_lemmas_list = [pred_lemma.lower().replace('ё', 'е')]

            true_lemma_norm = true_lemmas[i].lower().replace('ё', 'е')

            if not isinstance(pred_pos_list, list):
                pred_pos_list = [pred_pos_list] if pred_pos_list else ['unknown']
            
            pred_pos_ud_list = []
            for pos_oc in pred_pos_list:
                if pos_oc  == 'unknown':
                    word, pos_oc = find_closest_word_indexed(pred_lemma_norm, dictionary, index)
                    stem_lemma = stem_word(pred_lemma_norm)
                    new_lemma = add_ending_to_stem(stem_lemma, pos_oc)
                    pred_lemmas_list = [new_lemma]
                if pos_oc in mapping:
                    possible_tags = mapping[pos_oc]
                    if isinstance(possible_tags, list):
                        pred_pos_ud_list.extend(possible_tags)
                    else:
                        pred_pos_ud_list.append(possible_tags)
                else:
                    pred_pos_ud_list.append('X')
            
            pos_correct = true_tags_ud[i] in pred_pos_ud_list
            lemma_correct = true_lemma_norm in pred_lemmas_list
            
            if pos_correct:
                correct_pos += 1
            else:
                pos_errors.append({
                    'word': word,
                    'true_pos': true_tags_ud[i],
                    'pred_pos_list': pred_pos_ud_list,
                    'pred_pos_oc': pred_pos_list
                })
                true_tag = true_tags_ud[i]
                pos_error_by_tag[true_tag] = pos_error_by_tag.get(true_tag, 0) + 1
            
            if lemma_correct:
                correct_lemma += 1
            else:
                lemma_errors.append({
                    'word': word,
                    'true_lemma': true_lemma_norm,
                    'pred_lemmas': pred_lemmas_list,
                    'true_pos': true_tags_ud[i],
                    'pred_pos': pred_pos_ud_list
                })
            
            if pos_correct and lemma_correct:
                correct_both += 1
        print(j)
        j+=1
            
    pos_accuracy = correct_pos / total_tokens if total_tokens > 0 else 0
    lemma_accuracy = correct_lemma / total_tokens if total_tokens > 0 else 0
    both_accuracy = correct_both / total_tokens if total_tokens > 0 else 0

    print(f"Всего токенов: {total_tokens}")
    print(f"\nТочность тегирования (POS): {pos_accuracy:.2%}")
    print(f" Точность лемматизации:      {lemma_accuracy:.2%}")
    print(f" Полная точность (лемма+POS): {both_accuracy:.2%}")

    if pos_errors:
        total_pos_errors = len(pos_errors)
        for true_tag, error_count in sorted(pos_error_by_tag.items(), key=lambda x: x[1], reverse=True):
            tag_error_percent = error_count / total_pos_errors * 100
            print(f"  {true_tag:6}: {error_count:5d} ошибок ({tag_error_percent:5.1f}% от всех ошибок)")

    if lemma_errors:
        pos_with_errors = {}
        for err in lemma_errors:
            pos = err['true_pos']
            pos_with_errors[pos] = pos_with_errors.get(pos, 0) + 1
        
        print("\nОшибки лемматизации по частям речи:")
        for pos, count in sorted(pos_with_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pos}: {count} ошибок ({count/len(lemma_errors)*100:.1f}%)")
    
    return {
        'pos_accuracy': pos_accuracy,
        'lemma_accuracy': lemma_accuracy,
        'both_accuracy': both_accuracy,
        'total_tokens': total_tokens,
        'lemma_errors': len(lemma_errors)
    }

def tokenize():
    sentense_of_tokens = []
    while True:
        sentense = input()
        if sentense == '':
            break
        one_sentense = [token.replace("Ё", "е").replace("ё", "е") for token in re.split(r'[,.\n?! ]+', sentense) if token]
        sentense_of_tokens.append(one_sentense)
    return sentense_of_tokens


def load_dictionary(filename = "dictionary"):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def search_in_dict(word, dictionary):
    word_lower = word.lower()
    word_upper = word.upper()
    if word_upper in dictionary["lemmas"]:
        return word_upper.lower(), dictionary["lemmas"][word_upper]
    if word_lower in dictionary["wordforms"]:
        elements_from_dict = dictionary["wordforms"][word_lower]
        return elements_from_dict["lemma"], elements_from_dict["pos"]
    return None, None
    
def lemmatize_and_tagging(sentense_of_tokens, dictionary, index):
    result_sentences=[]
    for sentense in range(0, len(sentense_of_tokens)):
        formatted_tokens = []
        for token in range(0, len(sentense_of_tokens[sentense])):
            word = sentense_of_tokens[sentense][token]
            lemma, pos = search_in_dict(word, dictionary)
            if lemma:
                lemma = normalize_verb_lemma(lemma, pos, dictionary, RULES_INFINITIVE)
                formatted_tokens.append(f"{sentense_of_tokens[sentense][token]}{{{lemma.lower()}={pos}}}")
            else:
                _, pos = find_closest_word_indexed(word, dictionary, index)
                stem_lemma = stem_word(word)
                new_lemma = add_ending_to_stem(stem_lemma, pos)
                formatted_tokens.append(f"{sentense_of_tokens[sentense][token]}{{{new_lemma}={pos}}}") 
        result_sentences.append(" ".join(formatted_tokens))
    return result_sentences

def main():
    dictionary = load_dictionary()
    index = build_index(dictionary)
    while True:
        print("Введите exit для выхода, test для теста или text для ввода текста:")
        action = input()
        if action == "exit":
            break
        elif action == "text":
            sentense_of_tokens = tokenize()
            result_sentences = lemmatize_and_tagging(sentense_of_tokens, dictionary, index)
            print(result_sentences)
        elif action == "test":
            test_data = load_syntagrus_data('ru_syntagrus-ud-test.conllu')
            evaluate_on_syntagrus(test_data, dictionary, OPEN_CORPORA_TO_UD, index)
        else:
            print("Проверьте ввод")

if __name__ == '__main__':
    main()