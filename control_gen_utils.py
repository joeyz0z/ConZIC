import numpy as np
import torch
import torch.nn.functional as F
import random
from utils import get_init_text, update_token_mask
from sentiments_classifer import batch_texts_POS_Sentiments_analysis
from POS_classifier import batch_texts_POS_analysis

import time


def generate_caption_step(out, gen_idx, mask, temperature=None, top_k=0):
    """ Generate a word from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    probs *= (mask)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)

    return top_k_probs, top_k_ids

def sentiment_sequential_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                          max_len=15, top_k=0,temperature=None, alpha=0.7,beta=1,
                          max_iters=20,batch_size=1,
                         verbose=True,gamma=5, ctl_signal="positive"):
    """ Generate one word at a time, in L->R order """
    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer,prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list = []
    for iter_num in range(max_iters):
        for ii in range(max_len):
            token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
            inp[:,seed_len + ii] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii,mask=token_mask, top_k=top_k, temperature=temperature)
            topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            topk_inp[:,:,ii + seed_len] = idxs_ 
            repeats = ((idxs_[:,:, None] == topk_inp).float().sum(2) - 1)
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            sentiment_probs_batch, sentiment_scores_batch, pos_tags, wordnet_pos_tags = batch_texts_POS_Sentiments_analysis(
                    batch_text_list, 1, topk_inp.device, sentiment_ctl=ctl_signal,  batch_size_image = batch_size)
            clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
            final_score = alpha * probs + beta * clip_score + gamma * sentiment_probs_batch + 0.1 * (1-torch.exp(repeats))
            best_clip_id = final_score.argmax(dim=1).view(-1,1)
            inp[:,seed_len + ii] = idxs_.gather(1, best_clip_id).squeeze(-1)
            current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
            current_senti_score = sentiment_scores_batch.gather(1, best_clip_id).squeeze(-1)
            clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()
            senti_score_sequence_batch = current_senti_score.cpu().detach().numpy().tolist()
        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)            
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]}, clip score {clip_score_sequence_batch[jj]:.3f}"
                    f", ctl score {senti_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)

    return gen_texts_list, clip_score_sequence

def sentiment_shuffle_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                          max_len=15, top_k=0,temperature=None, alpha=0.7,beta=1,
                          max_iters=20,batch_size=1,
                           verbose=True,gamma=5, ctl_signal="positive"):
    """ Generate one word at a time, in random generation order """
    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer,prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    inp = torch.tensor(batch).to(image_embeds.device)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    random_lst = list(range(max_len))
    random.shuffle(random_lst)
    logger.info(f"Order_list:{random_lst}")
    gen_texts_list = []
    for iter_num in range(max_iters):
        for ii in random_lst:
            token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
            inp[:,seed_len + ii] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii,mask=token_mask, top_k=top_k, temperature=temperature)
            topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            topk_inp[:,:,ii + seed_len] = idxs_
            repeats = ((idxs_[:,:, None] == topk_inp).float().sum(2) - 1)
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            sentiment_probs_batch, sentiment_scores_batch, pos_tags, wordnet_pos_tags = batch_texts_POS_Sentiments_analysis(
                    batch_text_list, 1, topk_inp.device, sentiment_ctl=ctl_signal,  batch_size_image = batch_size) 
            clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
            final_score = alpha * probs + beta * clip_score + gamma * sentiment_probs_batch + 0.1 * (1-torch.exp(repeats))
            best_clip_id = final_score.argmax(dim=1).view(-1,1)
            inp[:,seed_len + ii] = idxs_.gather(1, best_clip_id).squeeze(-1)
            current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
            current_senti_score = sentiment_scores_batch.gather(1, best_clip_id).squeeze(-1)
            clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()
            senti_score_sequence_batch = current_senti_score.cpu().detach().numpy().tolist() 
        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)            
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]}, clip score {clip_score_sequence_batch[jj]:.3f}"
                    f", ctl score {senti_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)
    return gen_texts_list, clip_score_sequence

def POS_sequential_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                          max_len=15, top_k=0,temperature=None, alpha=0.7,beta=1,gamma=0.1,
                          max_iters=20,batch_size=1,ctl_signal=["DET"],
                          verbose=True):
    """ Generate one word at a time, in L->R order """

    seed_len = len(prompt.split())+1
    templete = False
    logger.info(ctl_signal)
    batch = get_init_text(tokenizer,prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_ctl_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list= []
    for iter_num in range(max_iters):
        for ii in range(max_len):
            token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
            inp[:,seed_len + ii] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii,mask=token_mask, top_k=top_k, temperature=temperature)
            topk_inp = inp_.unsqueeze(1).repeat(1,top_k,1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            topk_inp[:,:,ii + seed_len] = idxs_
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            pos_tags, pos_scores = batch_texts_POS_analysis(batch_text_list, ctl_signal, device=idxs_.device)
            pos_scores_batch = pos_scores.view([batch_size, -1])
            pos_probs = torch.softmax(pos_scores_batch/0.1, dim=-1).to(idxs_.device)
            clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
            final_score = alpha * probs + beta * clip_score + gamma * pos_probs
            best_clip_id = final_score.argmax(dim=1).view(-1,1)
            inp[:,seed_len + ii] = idxs_.gather(1, best_clip_id).squeeze(-1)
            current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
            current_ctl_score = pos_scores_batch.gather(1,best_clip_id).squeeze(-1)
            be_clip_id_batch = best_clip_id.reshape(-1).cpu()
            pos_tags_sequence_batch = []
            for i in range(batch_size):
                pos_tags_sequence_batch.append(pos_tags[be_clip_id_batch[i]+i*top_k])
            clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()
            ctl_score_sequence_batch = current_ctl_score.cpu().detach().numpy().tolist()                     
        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)            
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_ctl_score_list[jj] = ctl_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]}, clip score {clip_score_sequence_batch[jj]:.3f}"
                    f", ctl score {ctl_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
                logger.info(pos_tags_sequence_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)
    return gen_texts_list, clip_score_sequence

def control_generate_caption(img_name, model, clip, tokenizer,image_instance,token_mask,logger,
                     prompt="", batch_size=10, max_len=25,
                    top_k=100, temperature=1.0, max_iter=500,alpha=0.7,beta=1,gamma=5,
                    ctl_type="sentiment", style_type="positive",pos_type=None,generate_order="sequential"):
    # controllable funcitions to call
    start_time = time.time()
    if ctl_type=="sentiment": # sentiment control
        if generate_order=="sequential":
            generate_texts, clip_scores = sentiment_sequential_generation(img_name, model, clip, tokenizer, image_instance, token_mask, prompt, logger,
                                     batch_size=batch_size, max_len=max_len, top_k=top_k,
                                     alpha=alpha,beta=beta,gamma=gamma,temperature=temperature,
                                    max_iters=max_iter, ctl_signal=style_type)
        else:
            generate_texts, clip_scores = sentiment_shuffle_generation(img_name, model, clip, tokenizer, image_instance,
                                                                       token_mask, prompt, logger,
                                                                       batch_size=batch_size, max_len=max_len,
                                                                       top_k=top_k,
                                                                       alpha=alpha, beta=beta, gamma=gamma,
                                                                       temperature=temperature,
                                                                       max_iters=max_iter,
                                                                       ctl_signal=style_type)

    else: # POS control
        generate_texts, clip_scores = POS_sequential_generation(img_name, model, clip, tokenizer, image_instance, token_mask, prompt, logger,
                                 batch_size=batch_size, max_len=max_len, top_k=top_k,
                                 alpha=alpha,beta=beta,gamma=gamma,temperature=temperature, ctl_signal=pos_type,
                                  max_iters=max_iter)

    logger.info("Finished in %.3fs" % (time.time() - start_time))
    final_caption = generate_texts[-2]
    best_caption = generate_texts[-1]
    for i in range(batch_size):
        logger.info(f"The {i+1}-th image: {img_name[i]}")
        logger.info(f"final caption: {final_caption[i]}")
        logger.info(f"best caption: {best_caption[i]}")
    return generate_texts, clip_scores