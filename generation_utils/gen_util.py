import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import matplotlib.pyplot as plt
import os

def sample_visualizer(tokenizer, final_score, clip_score, probs, tokens, idxs_, current_clip_score, pic_id=0):
    pbs, ids = torch.topk(final_score, probs.shape[0])
    wds = tokenizer.convert_ids_to_tokens(idxs_[ids])

    bert_pbs, bert_ids = torch.topk(probs, probs.shape[0])
    bert_wds = tokenizer.convert_ids_to_tokens(idxs_[bert_ids])

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.suptitle(tokenizer.decode(tokens) + f'\n CLIPscore:{current_clip_score.cpu().detach().item():.3f}', fontsize=8)
    axes1 = axes[0]
    cprobs = clip_score[ids]
    axes1.bar(range(pbs.shape[0]), cprobs.cpu().detach().numpy())
    axes1.set_title('CLIP')

    axes2 = axes[1]
    wprobs = probs[ids]
    axes2.bar(range(pbs.shape[0]), wprobs.cpu().detach().numpy())
    axes2.set_title('BERT')
    axes2.annotate("TopK:", xy=(100, 0.65))
    for wid in range(15):
        axes2.annotate(bert_wds[wid], xy=(100, 0.6 - 0.03 * wid))

    axes3 = axes[2]
    axes3.bar(range(pbs.shape[0]), pbs.cpu().detach().numpy())
    axes3.set_ylim(0, 0.7)
    axes3.set_title('FINAL')
    axes3.annotate("TopK:", xy=(100, 0.65))
    for wid in range(15):
        axes3.annotate(wds[wid], xy=(100, 0.6 - 0.03 * wid))
    if os.path.exists("logger/demo")==False:
        os.makedirs("logger/demo")
    plt.savefig(f"logger/demo/{pic_id:03d}.jpg", dpi=300)
    pic_id += 1
    plt.close()

def get_init_text(tokenizer, seed_text, max_len, batch_size=1):
    """ Get initial sentence by padding seed_text with [mask] words to max_len """
    text = seed_text + tokenizer.mask_token * max_len
    ids = tokenizer.encode(text)
    batch = [ids] * batch_size
    return batch

def update_token_mask(tokenizer, token_mask, max_len, index):
    """ '.'(full stop) is only allowed in the last token position """
    if index == max_len - 1:
        token_mask[:, tokenizer.vocab['.']] = 1
    else:
        token_mask[:, tokenizer.vocab['.']] = 0
    return token_mask

def generate_step(out, gen_idx,  temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def generate_caption_step(out, gen_idx, mask, extend_ids=None,temperature=None, top_k=100):
    # out, gen_idx=seed_len + ii, mask=token_mask, top_k=top_k, temperature=temperature
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size (batch_size, seq_len, vocab_size)
        - gen_idx (int): location for which to generate for
        - mask (torch.Tensor): (1, vocab_size)
        - extend_ids: (batch_size, extend_len)
        - top_k (int): candidate k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    probs *= (mask)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)
    if extend_ids is not None:
        # Need to be optimize when extend_ids in top_k_ids
        top_k_probs = torch.cat((top_k_probs, torch.gather(probs,dim=-1,index=extend_ids)),dim=-1)
        top_k_ids = torch.cat((top_k_ids, extend_ids),dim=-1)

    return top_k_probs, top_k_ids

def fixed_generation(img_name, model, clip, tokenizer, image_instance,token_mask, prompt, logger,
                          max_len=15, top_k=100,temperature=None, alpha=0.7,beta=1,
                          max_iters=20, batch_size=1, stable_replace=False, verbose=True,visualize=False, shuffle=False):
    """ Generate one word at a time, in fixed order """

    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer, prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list = []
    order_lst = list(range(max_len))
    pic_id = 0
    if shuffle==True:
        random.shuffle(order_lst)
        logger.info(f"Generation order: {order_lst}")
    for iter_num in range(max_iters):
        for ii in order_lst:
            token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
            if stable_replace==True:
                old_ids = inp[:,seed_len + ii][:,None].clone()
            else:
                old_ids = None
            inp[:,seed_len + ii] = tokenizer.mask_token_id
            inp_ = inp.clone().detach()
            out = model(inp).logits
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii, mask=token_mask, extend_ids=old_ids, top_k=top_k, temperature=temperature)
            topk_inp = inp_.unsqueeze(1).repeat(1,idxs.shape[1],1)
            idxs_ = (idxs * token_mask[0][idxs]).long()
            topk_inp[:,:,ii + seed_len] = idxs_
            topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
            batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
            clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
            final_score = alpha * probs + beta * clip_score

            best_clip_id = final_score.argmax(dim=1).view(-1,1)
            inp[:,seed_len + ii] = idxs_.gather(1, best_clip_id).squeeze(-1)
            current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
            clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()
            if verbose and visualize:
                sample_visualizer(tokenizer, final_score[0], clip_score[0], probs[0], inp[0], idxs_[0], current_clip_score[0], pic_id)
                pic_id += 1
        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]},"
                            f"clip score {clip_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)

    return gen_texts_list, clip_score_sequence

def span_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                          max_len=15, top_k=0,temperature=None, alpha=0.7,beta=1,
                          max_iters=20,batch_size=1,stable_replace=False,verbose=True):
    """ Generate multiple words at a time (span generation), in L->R order """
    seed_len = len(prompt.split())+1
    span_len = 2
    batch = get_init_text(tokenizer,prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list= []
    for iter_num in range(max_iters):
        for span_start in range(0,max_len,span_len):
            span_end = min(span_start+span_len,max_len)
            if stable_replace==True:
                old_ids = inp[:,seed_len + span_start: seed_len + span_end].clone()
            else:
                old_ids = None
            inp[:,seed_len + span_start: seed_len + span_end] = tokenizer.mask_token_id
            out = model(inp).logits
            for ii in range(span_start,span_end):
                token_mask = update_token_mask(tokenizer, token_mask, max_len, ii)
                inp_ = inp.clone().detach()
                probs, idxs = generate_caption_step(out, gen_idx=seed_len + ii, mask=token_mask, top_k=top_k,
                                                    extend_ids=old_ids[:, ii - span_start][:, None], temperature=temperature)
                topk_inp = inp_.unsqueeze(1).repeat(1,idxs.shape[1],1)
                idxs_ = (idxs * token_mask[0][idxs]).long()
                topk_inp[:,:,ii + seed_len] = idxs_ 
                topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
                batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
                clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
                final_score = alpha * probs + beta * clip_score
                best_clip_id = final_score.argmax(dim=1).view(-1,1)
                inp[:,seed_len + ii] = idxs_.gather(1, best_clip_id).squeeze(-1)
                current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
                clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()                
        if verbose and np.mod(iter_num + 1, 1) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)
            for jj in range(batch_size):
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text_batch[jj]
                logger.info(f"iter {iter_num + 1}, The {jj+1}-th image: {img_name[jj]},"
                            f"clip score {clip_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])
        gen_texts_list.append(cur_text_batch)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)
    return gen_texts_list, clip_score_sequence

def random_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                    max_len=15, top_k=0, temperature=None,alpha=0.7,beta=2,max_iters=300,print_every=10,batch_size=1,stable_replace=False,verbose=True):
    """ Generate for one random position at a timestep"""

    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer, prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list = []
    for ii in range(max_iters):
        kk = np.random.randint(0, max_len)
        token_mask = update_token_mask(tokenizer, token_mask, max_len, kk)
        if stable_replace == True:
            old_ids = inp[:, seed_len + kk][:, None].clone()
        else:
            old_ids = None
        inp[:,seed_len + kk] = tokenizer.mask_token_id
        inp_ = inp.clone().detach()
        out = model(inp).logits
        probs, idxs = generate_caption_step(out,gen_idx=seed_len + kk,mask=token_mask, extend_ids=old_ids,top_k=top_k, temperature=temperature)
        topk_inp = inp_.unsqueeze(1).repeat(1,idxs.shape[1],1)
        idxs_ = (idxs * token_mask[0][idxs]).long()
        topk_inp[:,:,kk + seed_len] = idxs_ 
        topk_inp_batch = topk_inp.view(-1,topk_inp.shape[-1])
        batch_text_list= tokenizer.batch_decode(topk_inp_batch , skip_special_tokens=True)
        clip_score, clip_ref = clip.compute_image_text_similarity_via_raw_text(image_embeds, batch_text_list)
        final_score = alpha * probs + beta * clip_score
        best_clip_id = final_score.argmax(dim=1).view(-1,1)
        inp[:,seed_len + kk] = idxs_.gather(1, best_clip_id).squeeze(-1)
        current_clip_score = clip_ref.gather(1,best_clip_id).squeeze(-1)
        clip_score_sequence_batch = current_clip_score.cpu().detach().numpy().tolist()        
        cur_text_batch= tokenizer.batch_decode(inp,skip_special_tokens=True)
        for jj in range(batch_size):
            if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                best_caption_list[jj] = cur_text_batch[jj]
        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print_batch = tokenizer.batch_decode(inp)
            for jj in range(batch_size):                
                logger.info(f"iter {ii + 1}, The {jj+1}-th image: {img_name[jj]},"
                            f"clip score {clip_score_sequence_batch[jj]:.3f}: "+ for_print_batch[jj])                
            gen_texts_list.append(cur_text_batch)
            clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)

    return gen_texts_list, clip_score_sequence

def parallel_generation(img_name, model, clip, tokenizer,image_instance,token_mask, prompt, logger,
                        max_len=15, top_k=0, temperature=None,  alpha=0.1, beta=1,
                        max_iters=300,batch_size=1,print_every=1,stable_replace=False, verbose=True):
    """ Generate for all positions at a time step """
    seed_len = len(prompt.split())+1
    batch = get_init_text(tokenizer,prompt, max_len, batch_size)
    image_embeds = clip.compute_image_representation_from_image_instance(image_instance)
    clip_score_sequence = []
    best_clip_score_list = [0] * batch_size
    best_caption_list = ['None'] * batch_size
    inp = torch.tensor(batch).to(image_embeds.device)
    gen_texts_list = []
    for ii in range(max_iters):
        inp_ = inp.clone().detach()
        out = model(inp).logits
        gen_texts = []
        for kk in range(max_len):
            probs, idxs = generate_caption_step(out, gen_idx=seed_len + kk,mask=token_mask, top_k=top_k, temperature=temperature)
            clip_score_sequence_batch = []
            for jj in range(batch_size):
                topk_inp = inp_.unsqueeze(0).repeat(top_k,1,1)
                topk_inp[:, jj, ii + seed_len] = (idxs[jj] * token_mask[0][idxs[jj]]).long()
                batch_text_list = tokenizer.batch_decode(topk_inp[:,jj,:], skip_special_tokens=True)
                single_image_embeds = image_embeds[jj].unsqueeze(0)
                clip_score,clip_ref = clip.compute_image_text_similarity_via_raw_text(single_image_embeds, batch_text_list)
                final_score = alpha * probs[jj,:] + beta * clip_score
                best_clip_id = final_score.argmax()
                inp[jj][seed_len + kk] = idxs[jj][best_clip_id]
                current_clip_score = clip_ref[0][best_clip_id]
                clip_score_sequence_batch.append(current_clip_score.cpu().item())
        if verbose and np.mod(ii, 1) == 0:
            for jj in range(batch_size):
                for_print = tokenizer.decode(inp[jj])
                cur_text = tokenizer.decode(inp[jj],skip_special_tokens=True)
                if best_clip_score_list[jj] < clip_score_sequence_batch[jj]:
                    best_clip_score_list[jj] = clip_score_sequence_batch[jj]
                    best_caption_list[jj] = cur_text
                gen_texts.append(cur_text)
                logger.info(f"iter {ii + 1}, The {jj+1}-th image: {img_name[jj]}, clip score {clip_score_sequence_batch[jj]:.3f}: "+ for_print)
        gen_texts_list.append(gen_texts)
        clip_score_sequence.append(clip_score_sequence_batch)
    gen_texts_list.append(best_caption_list)
    clip_score_sequence.append(best_clip_score_list)
    return gen_texts_list, clip_score_sequence

def generate_caption(img_name, model, clip, tokenizer,image_instance,token_mask,logger,
                     prompt="", batch_size=1, max_len=15,
                     top_k=100, temperature=1.0, max_iter=500,alpha=0.7,beta=1,
                     generate_order="sequential", stable_replace=False):
    # main generation functions to call
    start_time = time.time()
    if len(img_name) != batch_size: # last_batch < batch_size
        batch_size = len(img_name)

    if generate_order=="sequential":
        generate_texts, clip_scores = fixed_generation(img_name, model, clip, tokenizer,image_instance,token_mask,prompt, logger,
                                 batch_size=batch_size, max_len=max_len, top_k=top_k,
                                 alpha=alpha,beta=beta,temperature=temperature,max_iters=max_iter,
                                 stable_replace=stable_replace,shuffle=False)

    elif generate_order=="shuffle":
        generate_texts, clip_scores = fixed_generation(img_name, model, clip, tokenizer,image_instance,token_mask,prompt, logger,
                                 batch_size=batch_size, max_len=max_len, top_k=top_k,
                                 alpha=alpha,beta=beta,temperature=temperature,max_iters=max_iter,
                                 stable_replace=stable_replace,shuffle=True)

    elif generate_order=="random":
        max_iter *= max_len
        print_every = max_len
        generate_texts, clip_scores = random_generation(img_name, model, clip, tokenizer,image_instance,token_mask,prompt,logger,
                              max_len=max_len, top_k=top_k,alpha=alpha,beta=beta,print_every=print_every,
                               temperature=temperature, batch_size=batch_size, max_iters=max_iter,stable_replace=stable_replace,verbose=True)

    elif generate_order=="span":
        max_iter = max_iter
        generate_texts, clip_scores = span_generation(img_name, model, clip, tokenizer, image_instance, token_mask, prompt, logger,
                                 batch_size=batch_size, max_len=max_len, top_k=top_k,
                                 alpha=alpha,beta=beta,temperature=temperature, max_iters=max_iter,stable_replace=stable_replace)

    elif generate_order=="parallel":
        assert("Not implemented now!")
        generate_texts, clip_scores = parallel_generation(img_name, model, clip, tokenizer,image_instance,token_mask,prompt,  logger,
                               max_len=max_len, temperature=temperature, top_k=top_k, alpha=alpha,beta=beta,
                                max_iters=max_iter, batch_size=batch_size,stable_replace=stable_replace, verbose=True)

    logger.info("Finished in %.3fs" % (time.time() - start_time))
    final_caption = generate_texts[-2]
    best_caption = generate_texts[-1]
    for i in range(batch_size):
        logger.info(f"The {i+1}-th image: {img_name[i]}")
        logger.info(f"final caption: {final_caption[i]}")
        logger.info(f"best caption: {best_caption[i]}")
    return generate_texts, clip_scores