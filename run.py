import os
import time
import json
import torch
from torch.utils.data import DataLoader

from args import get_args
from utils import create_logger,set_seed
from vl_models.clip import CLIP
from generation_utils.gen_util import generate_caption
from generation_utils.control_gen_util import control_generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataset.ImgDataset import Imgdata, collate_img

def run_caption(args, img_name, img_pil_list, lm_model, lm_tokenizer, clip, token_mask, logger, all_results):
    image_instance = img_pil_list
    gen_texts, clip_scores = generate_caption(img_name, lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                top_k=args.candidate_k, temperature=args.lm_temperature,
                                max_iter=args.num_iterations, alpha=args.alpha,beta=args.beta,
                                generate_order = args.order, stable_replace=args.stable_replace)
    for iter_id, gen_text_list in enumerate(gen_texts):
        for jj in range(len(gen_text_list)):
            image_id = img_name[jj].split(".")[0]
            if all_results[iter_id]==None:
                all_results[iter_id] = {image_id: gen_text_list[jj]}
            else:
                all_results[iter_id][image_id] = gen_text_list[jj]
    return all_results

def run_control(run_type, args, img_name, img_pil_list, lm_model, lm_tokenizer, clip, token_mask, logger, all_results):

    image_instance = img_pil_list
    gen_texts, clip_scores = control_generate_caption(img_name, lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                top_k=args.candidate_k, temperature=args.lm_temperature,
                                max_iter=args.num_iterations, alpha=args.alpha,
                                beta=args.beta, gamma=args.gamma,stable_replace=args.stable_replace,
                                ctl_type = args.control_type, style_type=args.sentiment_type,pos_type=args.pos_type, generate_order=args.order)

    for iter_id, gen_text_list in enumerate(gen_texts):
        for jj in range(len(gen_text_list)):
            image_id = img_name[jj].split(".")[0]
            if all_results[iter_id]==None:
                all_results[iter_id] = {image_id: gen_text_list[jj]}
            else:
                all_results[iter_id][image_id] = gen_text_list[jj]
    return all_results

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    run_type = "caption" if args.run_type=="caption" else args.control_type
    if run_type=="sentiment":
        run_type = args.sentiment_type

    if os.path.exists("logger")== False:
        os.mkdir("logger")
    logger = create_logger(
        "logger",'{}_{}_len{}_topk{}_alpha{}_beta{}_gamma{}_lmtemp{}_{}.log'.format(
        run_type, args.order,args.sentence_len,
        args.candidate_k, args.alpha,args.beta,args.gamma,args.lm_temperature,
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    logger.info(f"Generating order:{args.order}")
    logger.info(f"Run type:{run_type}")
    logger.info(args)

    # Load pre-trained model (weights)
    lm_model = AutoModelForMaskedLM.from_pretrained(args.lm_model)
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model.eval()
    clip = CLIP(args.match_model)
    clip.eval()

    lm_model = lm_model.to(args.device)
    clip = clip.to(args.device)

    ## Remove stop words, token mask
    with open(args.stop_words_path,'r',encoding='utf-8') as stop_words_file:
        stop_words = stop_words_file.readlines()
        stop_words_ = [stop_word.rstrip('\n') for stop_word in stop_words]
        stop_words_ += args.add_extra_stopwords
        stop_ids = lm_tokenizer.convert_tokens_to_ids(stop_words_)
        token_mask = torch.ones((1,lm_tokenizer.vocab_size))
        for stop_id in stop_ids:
            token_mask[0,stop_id]=0
        token_mask = token_mask.to(args.device)

    # Dataset
    img_dir = args.caption_img_path
    img_data = Imgdata(img_dir)
    train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=False)

    for sample_id in range(args.samples_num):
        all_results = [None] * (args.num_iterations+1)
        logger.info(f"Sample {sample_id+1}: ")
        for batch_idx, (img_batch_pil_list, name_batch_list) in enumerate(train_loader):
            logger.info(f"The {batch_idx+1}-th batch:")
            if args.run_type == 'caption':
                all_results = run_caption(args, name_batch_list, img_batch_pil_list, lm_model, lm_tokenizer, clip, token_mask, logger, all_results)
            elif args.run_type == 'controllable':
                all_results = run_control(run_type, args, name_batch_list, img_batch_pil_list,lm_model, lm_tokenizer, clip, token_mask, logger, all_results)
            else:
                raise Exception('run_type must be caption or controllable!')

        # Save results
        if args.run_type == 'caption':

            save_dir = "results/caption_%s_len%d_topk%d_alpha%.3f_beta%.3f_gamma%.3f_lmTemp%.3f/sample_%d" % (
            args.order,args.sentence_len, args.candidate_k, args.alpha, args.beta,args.gamma,args.lm_temperature,sample_id)
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            for iter_id in range(len(all_results)):
                if iter_id!=len(all_results)-1:
                    cur_json_file = os.path.join(save_dir,f"iter_{iter_id}.json")
                    with open(cur_json_file,'w') as _json:
                        json.dump(all_results[iter_id], _json)
                else:
                    cur_json_file = os.path.join(save_dir,f"best_clipscore.json")
                    with open(cur_json_file,'w') as _json:
                        json.dump(all_results[iter_id], _json)
        elif args.run_type == 'controllable':
            save_dir = "results/%s_%s_len%d_topk%d_alpha%.3f_beta%.3f_gamma%.3f_lmTemp%.3f/sample_%d" % (
            run_type,args.order,args.sentence_len, args.candidate_k, args.alpha, args.beta,args.gamma,args.lm_temperature, sample_id)
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            for iter_id in range(len(all_results)):
                if iter_id!=len(all_results)-1:
                    cur_json_file = os.path.join(save_dir,f"iter_{iter_id}.json")
                    with open(cur_json_file,'w') as _json:
                        json.dump(all_results[iter_id], _json)
                else:
                    cur_json_file = os.path.join(save_dir,f"best_clipscore.json")
                    with open(cur_json_file,'w') as _json:
                        json.dump(all_results[iter_id], _json)        


