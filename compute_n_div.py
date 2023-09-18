from nltk.tokenize import word_tokenize
from collections import defaultdict
import json

def calc_diversity(predicts,vocab):
    tokens = [0.0, 0.0]
    types = [defaultdict(int), defaultdict(int)]
    for gg in predicts:
        g = word_tokenize(gg.lower())
        # g = gg.rstrip().lower().rstrip(".").split()
        for word in g:
            if word not in vocab:
                vocab.append(word)
        for n in range(2):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0]
    div2 = len(types[1].keys())/tokens[1]
    return [div1, div2], vocab

def calc_vocab_num(predicts):
    vocab = []
    for sentence in predicts:
        g = word_tokenize(sentence.lower())
        for word in g:
            if word not in vocab:
                vocab.append(word)
    return vocab

div1 = 0
div2 = 0
json_path = "diversity_formal.json"

vocab = []
with open(json_path,"r") as cur_json_file:
    cur_res = json.load(cur_json_file)
    for item in cur_res:
        div_n, vocab = calc_diversity(item["captions"],vocab)
        div1 += div_n[0]
        div2 += div_n[1]
    div1 /= len(cur_res)
    div2 /= len(cur_res)
with open("stop_words.txt",'r') as stop_word_file:
    stop_words = stop_word_file.readlines()
    stop_words = [word.rstrip() for word in stop_words]
    vocab = [word for word in vocab if (word not in stop_words and "unused" not in word)]
print("vocab_len:",len(set(vocab)))
print("div_1:",div1)
print("div_2:",div2)

