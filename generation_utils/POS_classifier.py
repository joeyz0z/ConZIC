from nltk.tokenize import word_tokenize
from nltk import pos_tag
import torch
import json

def batch_texts_POS_analysis(batch_texts, pos_templete, device="cuda"):
    batch_size = len(batch_texts)
    pos_tags = []
    pos_scores = torch.zeros(batch_size)

    for b_id in range(batch_size):
        text = batch_texts[b_id]
        words = word_tokenize(text)
        word_tag = pos_tag(words, tagset="universal")
        res_tag = [tag[1] for tag in word_tag]
        total_num = len(pos_templete)
        correct = 0
        if len(res_tag) <= total_num:
            cur_tag = res_tag + [""] * (len(pos_templete)-len(res_tag))
        else:
            cur_tag = res_tag[:total_num]
        for word_id in range(len(cur_tag)):
            if pos_templete[word_id]=="":
                correct += 1
            elif cur_tag[word_id] in pos_templete[word_id]:
                correct +=1
        acc = correct/total_num
        pos_tags.append(res_tag)
        pos_scores[b_id] = acc

    return pos_tags, pos_scores

def text_POS_analysis(text):
    words = word_tokenize(text)
    word_tag = pos_tag(words, tagset="universal")
    res_tag = [tag[1] for tag in word_tag]

    return res_tag

if __name__=="__main__":
    batch_texts = ["A cat sitting in the bed.",
                   "Two men in a nice hotel room one playing a video game with a remote control.",
                   "The man sitting in the chair feels like an invisible,dead man."]
    pos_templete = ['DET', 'NOUN', 'ADP', 'ADJ', 'NOUN', '.', 'NOUN', 'CONJ', 'NOUN', 'ADP', 'PRON', '.']

    batch_texts_POS_analysis(batch_texts, pos_templete, device="cuda")
    cur_path = "iter_15.json"
    all_caption = []

    with open(cur_path, "r") as cur_json_file:
        all_res = list(json.load(cur_json_file).values())
        for res in all_res:
            if isinstance(res, list):
                all_caption += res
            else:
                all_caption.append(res)
        pos_tags, pos_scores = batch_texts_POS_analysis(all_caption, pos_templete, device="cuda")
        word_id = 12
        pos_dict = {"ADJ": 0, "ADP": 0, "ADV": 0,
                    "CONJ": 0, "DET": 0, "NOUN": 0,"X":0,
                    "NUM": 0, "PRT": 0, "PRON": 0, "VERB": 0, ".": 0}
        for pos_tag in pos_tags:
            if word_id < len(pos_tag):
                pos_dict[pos_tag[word_id]] += 1
        print(1)




