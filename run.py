from utils import create_logger,set_seed
import os
import time
import argparse
import json
from PIL import Image
import torch

from clip.clip import CLIP
from gen_utils import generate_caption
from control_gen_utils import control_generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2, help = "support batch_size>1 currently.")
    parser.add_argument("--device", type=str,
                        default='cuda',choices=['cuda','cpu'])

    ## Generation and Controllable Type
    parser.add_argument('--run_type',
                        default='controllable',
                        nargs='?',
                        choices=['caption', 'controllable'])
    parser.add_argument('--prompt',
                        default='Image of a',type=str)
    parser.add_argument('--order',
                        default='shuffle',
                        nargs='?',
                        choices=['sequential', 'shuffle', 'span', 'random'],
                        help="Generation order of text")
    parser.add_argument('--control_type',
                        default='sentiment',
                        nargs='?',
                        choices=["sentiment","pos"],
                        help="which controllable task to conduct")
    parser.add_argument('--pos_type', type=list,
                        default=[['DET'], ['ADJ','NOUN'], ['NOUN'],
                                 ['VERB'], ['VERB'],['ADV'], ['ADP'],
                                 ['DET','NOUN'], ['NOUN'], ['NOUN','.'],
                                 ['.','NOUN'],['.','NOUN']],
                        help="predefined part-of-speech templete")
    parser.add_argument('--sentiment_type',
                        default="positive",
                        nargs='?',
                        choices=["positive", "negative"])
    parser.add_argument('--samples_num',
                        default=2,type=int)

    ## Hyperparameters
    parser.add_argument("--sentence_len", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.02, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=2.0, help="weight for image-matching degree")
    parser.add_argument("--gamma", type=float, default=5.0, help="weight for controllable degree")
    parser.add_argument("--lm_temperature", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=10, help="predefined iterations for Gibbs Sampling")

    ## Models and Paths
    parser.add_argument("--lm_model", type=str, default='bert-base-uncased',
                        help="Path to language model") # bert,roberta
    parser.add_argument("--match_model", type=str, default='clip-vit-base-patch32',
                        help="Path to Image-Text model")  # clip,align
    parser.add_argument("--caption_img_path", type=str, default='./examples/',
                        help="file path of images for captioning")
    parser.add_argument("--stop_words_path", type=str, default='stop_words.txt',
                        help="Path to stop_words.txt")
    parser.add_argument("--add_extra_stopwords", type=list, default=[],
                        help="you can add some extra stop words")

    args = parser.parse_args()

    return args

def run_caption(args, img_name, img_pil_list, lm_model, lm_tokenizer, clip, token_mask, logger, all_results):

    image_instance = img_pil_list
    gen_texts, clip_scores = generate_caption(img_name, lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                top_k=args.candidate_k, temperature=args.lm_temperature,
                                max_iter=args.num_iterations,alpha=args.alpha,beta=args.beta,
                                generate_order = args.order)
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
                                beta=args.beta, gamma=args.gamma,
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

    img_dir = args.caption_img_path

    class Imgdata(Dataset):
        def __init__(self, dir_path):
            self.dir_path = dir_path
            self.img_name_list = os.listdir(dir_path)

        def __getitem__(self, idx):
            img_name = self.img_name_list[idx]
            img_item_path = os.path.join(self.dir_path,img_name)
            img = Image.open(img_item_path).convert("RGB")
            return img, img_name
        def __len__(self):
            return len(self.img_name_list)
    
    def collate_img(batch_data):
        img_path_batch_list = list()
        name_batch_list = list()
        for unit in batch_data:
            img_path_batch_list.append(unit[0])
            name_batch_list.append(unit[1])
        return img_path_batch_list,name_batch_list
    
    img_data = Imgdata(img_dir)
    train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=True)

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

        if args.run_type == 'caption':
            # 保存结果
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


