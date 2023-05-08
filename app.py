from utils import create_logger, set_seed, format_output
import os
import time
import argparse
import json
from PIL import Image
import torch
import gradio as gr
import nltk

from clip.clip import CLIP
from gen_utils import generate_caption
from control_gen_utils import control_generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help = "Only supports batch_size=1 currently.")
    parser.add_argument("--device", type=str,
                        default='cuda',choices=['cuda','cpu'])

    ## Generation and Controllable Type
    parser.add_argument('--run_type',
                        default='caption',
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
    parser.add_argument("--num_iterations", type=int, default=1, help="predefined iterations for Gibbs Sampling")

    ## Models and Paths
    parser.add_argument("--lm_model", type=str, default='bert-base-uncased',
                        help="Path to language model") # bert,roberta
    parser.add_argument("--match_model", type=str, default='clip-vit-base-patch32',
                        help="Path to Image-Text model")  # clip,align
    parser.add_argument("--caption_img_path", type=str, default='./examples/girl.jpg',
                        help="file path of the image for captioning")
    parser.add_argument("--stop_words_path", type=str, default='stop_words.txt',
                        help="Path to stop_words.txt")
    parser.add_argument("--add_extra_stopwords", type=list, default=[],
                        help="you can add some extra stop words")

    args = parser.parse_args()

    return args

def run_caption(args, image, lm_model, lm_tokenizer, clip, token_mask, logger):
    FinalCaptionList = []
    BestCaptionList = []
    img_name = ['Your image']
    image_instance = image.convert("RGB")
    for sample_id in range(args.samples_num):
        logger.info(f"Sample {sample_id}: ")
        gen_texts, clip_scores = generate_caption(img_name, lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                  prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                  top_k=args.candidate_k, temperature=args.lm_temperature,
                                  max_iter=args.num_iterations,alpha=args.alpha,beta=args.beta,
                                  generate_order = args.order)
        FinalCaptionStr = "Sample {}: ".format(sample_id + 1) + gen_texts[-2][0]
        BestCaptionStr = "Sample {}: ".format(sample_id + 1) + gen_texts[-1][0]
        FinalCaptionList.append(FinalCaptionStr)
        BestCaptionList.append(BestCaptionStr)
    return FinalCaptionList, BestCaptionList


    
def run_control(run_type, args, image, lm_model, lm_tokenizer, clip, token_mask, logger):
    FinalCaptionList = []
    BestCaptionList = []
    img_name = ['Your image']
    image_instance = image.convert("RGB")
    for sample_id in range(args.samples_num):
        logger.info(f"Sample {sample_id}: ")
        gen_texts, clip_scores = control_generate_caption(img_name, lm_model, clip, lm_tokenizer, image_instance, token_mask, logger,
                                  prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
                                  top_k=args.candidate_k, temperature=args.lm_temperature,
                                  max_iter=args.num_iterations, alpha=args.alpha,
                                  beta=args.beta, gamma=args.gamma,
                                  ctl_type = args.control_type, style_type=args.sentiment_type,pos_type=args.pos_type, generate_order=args.order)
        FinalCaptionStr = "Sample {}: ".format(sample_id + 1) + gen_texts[-2][0]
        BestCaptionStr = "Sample {}: ".format(sample_id + 1) + gen_texts[-1][0]
        FinalCaptionList.append(FinalCaptionStr)
        BestCaptionList.append(BestCaptionStr)
    return FinalCaptionList, BestCaptionList

def Demo(RunType, ControlType, SentimentType, Order, Length, NumIterations, SamplesNum, Alpha, Beta, Gamma, Img):
    args = get_args()
    set_seed(args.seed)

    args.num_iterations = NumIterations
    args.sentence_len = Length
    args.run_type = RunType
    args.control_type = ControlType
    args.sentiment_type = SentimentType
    args.alpha = Alpha
    args.beta = Beta
    args.gamma = Gamma
    args.samples_num = SamplesNum
    args.order = Order 
    img = Img

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

    if args.run_type == 'caption':
        FinalCaption, BestCaption = run_caption(args, img, lm_model, lm_tokenizer, clip, token_mask, logger)
    elif args.run_type == 'controllable':
        FinalCaption, BestCaption = run_control(run_type, args, img, lm_model, lm_tokenizer, clip, token_mask, logger)
    else:
        raise Exception('run_type must be caption or controllable!')

    logger.handlers = []

    FinalCaptionFormat, BestCaptionFormat = format_output(SamplesNum, FinalCaption, BestCaption)
    return FinalCaptionFormat, BestCaptionFormat


def RunTypeChange(choice):
    if choice == "caption":
        return gr.update(visible=False)
    elif choice == "controllable":
        return gr.update(visible=True)


def ControlTypeChange(choice):
    if choice == "pos":
        return gr.update(visible=False)
    elif choice == "sentiment":
        return gr.update(visible=True)
    
with gr.Blocks() as demo:

    gr.Markdown("""
    # ConZIC
    ### Controllable Zero-shot Image Captioning by Sampling-Based Polishing
    """)

    with gr.Row():
        with gr.Column():
            RunType = gr.Radio(
                ["caption", "controllable"], value="caption", label="Run Type", info="Select the Run Type"
            )
            ControlType = gr.Radio(
                ["sentiment", "pos"], value="sentiment", label="Control Type", info="Select the Control Type",
                visible=False, interactive=True
            )
            SentimentType = gr.Radio(
                ["positive", "negative"], value="positive", label="Sentiment Type", info="Select the Sentiment Type",
                visible=False, interactive=True
            )
            Order = gr.Radio(
                ["sequential", "shuffle", "random"], value="shuffle", label="Order", info="Generation order of text"
            )

            RunType.change(fn = RunTypeChange, inputs = RunType, outputs = SentimentType)
            RunType.change(fn = RunTypeChange, inputs = RunType, outputs = ControlType)
            ControlType.change(fn = ControlTypeChange, inputs = ControlType, outputs = SentimentType)

            with gr.Row():
                Length = gr.Slider(
                    5, 15, value=10, label="Sentence Length", info="Choose betwen 5 and 15", step=1
                )
                NumIterations = gr.Slider(
                    1, 15, value=10, label="Num Iterations", info="predefined iterations for Gibbs Sampling", step=1
                )
            with gr.Row():
                SamplesNum = gr.Slider(
                    1, 5, value=2, label="Samples Num", step=1
                )
                Alpha = gr.Slider(
                    0, 1, value=0.02, label="Alpha", info="Weight for fluency", step=0.01
                )
            with gr.Row():
                Beta = gr.Slider(
                    1, 5, value=2, label="Beta", info="Weight for image-matching degree", step=0.5
                )
                Gamma = gr.Slider(
                    1, 10, value=5, label="Gamma", info="weight for controllable degree", step=0.5
                )
        with gr.Column():

            Img = gr.Image(label="Upload Picture", type = "pil")

            FinalCaption = gr.Textbox(label="Final Caption", lines=5, placeholder="Final Caption")
            BestCaption = gr.Textbox(label="Best Caption", lines=5, placeholder="Best Caption")
            with gr.Row():
                gen_button = gr.Button("Submit")
                clear_button = gr.Button("Reset")
    
    gen_button.click(
        fn = Demo, 
        inputs = [
            RunType, ControlType, SentimentType, Order, Length, NumIterations, SamplesNum, Alpha, Beta, Gamma, Img
        ],
        outputs = [
            FinalCaption, BestCaption
        ]
    )
    clear_button.click(
        fn = lambda : [gr.Radio.update(value = 'caption'), gr.Radio.update(value = 'pos'), gr.Radio.update(value = 'positive'),
            gr.Radio.update(value = 'shuffle'), gr.Slider.update(value = 10), gr.Slider.update(value = 10),
            gr.Slider.update(value = 2), gr.Slider.update(value = 0.02), gr.Slider.update(value = 2),
            gr.Slider.update(value = 5)
        ],
        inputs = [
        ],
        outputs = [
           RunType, ControlType, SentimentType, Order, Length, NumIterations, SamplesNum, Alpha, Beta, Gamma
        ]
    )
if __name__ == "__main__":

    # nltk.download('wordnet')
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('sentiwordnet')

    demo.launch()
