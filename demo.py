from utils import create_logger,set_seed
import os
import time
from PIL import Image
import torch

from args import get_args
from vl_models.clip import CLIP
from generation_utils.gen_util import generate_caption
from generation_utils.control_gen_util import control_generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer

def run_caption(args, image_path, lm_model, lm_tokenizer, clip, token_mask, logger):

    logger.info(f"Processing: {image_path}")
    image_instance = Image.open(image_path).convert("RGB")
    img_name = [image_path.split("/")[-1]]
    for sample_id in range(args.samples_num):
        logger.info(f"Sample {sample_id}: ")
        gen_texts, clip_scores = generate_caption(img_name,lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                  prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                  top_k=args.candidate_k, temperature=args.lm_temperature,
                                  max_iter=args.num_iterations,alpha=args.alpha,beta=args.beta,
                                  generate_order = args.order, stable_replace=args.stable_replace)

def run_control(run_type, args, image_path, lm_model, lm_tokenizer, clip, token_mask, logger):

    logger.info(f"Processing: {image_path}")
    image_instance = Image.open(image_path).convert("RGB")
    img_name = [image_path.split("/")[-1]]
    for sample_id in range(args.samples_num):
        logger.info(f"Sample {sample_id}: ")
        gen_texts, clip_scores = control_generate_caption(img_name,lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                  prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                  top_k=args.candidate_k, temperature=args.lm_temperature,
                                  max_iter=args.num_iterations, alpha=args.alpha,
                                  beta=args.beta, gamma=args.gamma,stable_replace=args.stable_replace,
                                  ctl_type = args.control_type, style_type=args.sentiment_type,pos_type=args.pos_type, generate_order=args.order)

if __name__ == "__main__":
    args = get_args()
    args.batch_size=1
    set_seed(args.seed)
    run_type = "caption" if args.run_type=="caption" else args.control_type
    if run_type=="sentiment":
        run_type = args.sentiment_type
    
    if os.path.exists("logger")== False:
        os.mkdir("logger")
    logger = create_logger(
        "logger",'demo_{}_{}_len{}_topk{}_alpha{}_beta{}_gamma{}_lmtemp{}_{}.log'.format(
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

    img_path = args.caption_img_path
    if args.run_type == 'caption':
        run_caption(args, img_path, lm_model, lm_tokenizer, clip, token_mask, logger)
    elif args.run_type == 'controllable':
        run_control(run_type, args, img_path, lm_model, lm_tokenizer, clip, token_mask, logger)
    else:
        raise Exception('run_type must be caption or controllable!')



