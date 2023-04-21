from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import sentiwordnet
import torch
import torch.nn.functional as F



def text_POS_Sentiments_analysis(text,sentiment_ctl=None):
    """
    id: 0,1,2,3,4
    pos:none,n,v,a,r
    """
    words = word_tokenize(text)

    word_tag = pos_tag(words)
    res_tag = [tag[1] for tag in word_tag]
    tag_map = {'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n', 'UH': 'n', \
               'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', \
               'JJ': 'a', 'JJR': 'a', 'JJS': 'a', \
               'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r', 'WRB': 'r'}

    word_tag = [(t[0], tag_map[t[1]]) if t[1] in tag_map else (t[0], '') for t in word_tag]

    wordnet_tag = [tag[1] for tag in word_tag]
    sentiment_synsets = [list(sentiwordnet.senti_synsets(t[0], t[1])) for t in word_tag]

    if sentiment_ctl is None:
        return 0, res_tag, wordnet_tag
    score = sum(sum([x.pos_score() - x.neg_score() for x in s]) / len(s) for s in sentiment_synsets if len(s) != 0)
    if sentiment_ctl=="negative":
        score = -score
    return score, res_tag, wordnet_tag

def batch_texts_POS_Sentiments_analysis(batch_texts, temperature,device,sentiment_ctl=None, batch_size_image=1):
    batch_size = len(batch_texts)
    senti_scores = torch.zeros(batch_size)
    pos_tags = []
    wordnet_pos_tags = []
    for b_id in range(batch_size):
        text = batch_texts[b_id]
        score, cur_tag, cur_word_tag = text_POS_Sentiments_analysis(text,sentiment_ctl=sentiment_ctl)
        senti_scores[b_id] = score
        pos_tags.append(cur_tag)
        wordnet_pos_tags.append(cur_word_tag)
    senti_scores_batch = senti_scores.view(batch_size_image, -1).to(device)
    senti_probs_batch = F.softmax(senti_scores_batch / temperature,dim=1).to(device)
    return senti_probs_batch, senti_scores_batch, pos_tags, wordnet_pos_tags



