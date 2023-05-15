import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help = "run.py batch_size>1, demo.py batch_size=1.")
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
                        default='sequential',
                        nargs='?',
                        choices=['sequential', 'shuffle', 'span', 'random'],
                        help="Generation order of text")
    parser.add_argument('--control_type',
                        default='pos',
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
                        default="negative",
                        nargs='?',
                        choices=["positive", "negative"])
    parser.add_argument('--samples_num',default=1,type=int)
    parser.add_argument('--stable_replace',default=True,type=bool)
    parser.add_argument('--add_detect_tokens',default=True,type=bool)

    ## Hyperparameters
    parser.add_argument("--sentence_len", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.02, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=2.0, help="weight for image-matching degree")
    parser.add_argument("--gamma", type=float, default=5.0, help="weight for controllable degree")
    parser.add_argument("--lm_temperature", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=10, help="predefined iterations for Gibbs Sampling")

    ## Models and Paths
    parser.add_argument("--lm_model", type=str, default='../../../HuggingFace/bert-base-uncased',
                        help="Path to language model") # bert,roberta
    parser.add_argument("--match_model", type=str, default='../../../HuggingFace/clip-vit-base-patch32',
                        help="Path to Image-Text model")  # vl_models,align
    parser.add_argument("--caption_img_path", type=str, default='./examples',
                        help="file path of images for captioning")
    parser.add_argument("--stop_words_path", type=str, default='stop_words.txt',
                        help="Path to stop_words.txt")
    parser.add_argument("--add_extra_stopwords", type=list, default=[],
                        help="you can add some extra stop words")

    args = parser.parse_args()

    return args