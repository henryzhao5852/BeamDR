from os.path import join
import sys
sys.path += ['../']
import argparse
import json
import os
import random
import numpy as np
import torch
from model.models import MSMarcoConfigDict, ALL_MODELS
import csv
#from utils.util import multi_file_process, numbered_byte_file_generator, EmbeddingCache
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_data_dir",
    default="/fs/clip-scratch/chen/hotpot_data/",
    type=str,
    help="The output data dir",
)
parser.add_argument(
    "--num",
    default=0,
    type=int,
    help="The output data dir",
)
parser.add_argument(
    "--model_type",
    default="dpr",
    type=str,
    help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
)
parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pre-trained model or shortcut name selected in the list: " +
    ", ".join(ALL_MODELS),
)
parser.add_argument(
    "--max_seq_length",
    default=256,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--wiki_dir",
    default='/fs/clip-scratch/chen/data/wikipedia_split/',
    type=str,
    help="location of the wiki corpus",
)
args = parser.parse_args()


def PassagePreprocessingFn(args, line, tokenizer):
    line_arr = list(csv.reader([line], delimiter='\t'))[0]
    if line_arr[0] == 'id':
        return bytearray()

    p_id = int(line_arr[0])
    text = line_arr[1]
    title = line_arr[2].replace('_', ' ')

    token_ids = tokenizer.encode(title, text_pair=text, add_special_tokens=True,
                                      max_length=args.max_seq_length,
                                      pad_to_max_length=False)

    seq_len = args.max_seq_length
    passage_len = len(token_ids)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [tokenizer.pad_token_id] * (seq_len - len(token_ids))
    if len(token_ids) > seq_len:
        token_ids = token_ids[0:seq_len]
        token_ids[-1] = tokenizer.sep_token_id

    if p_id < 5:
        a = np.array(token_ids, np.int32)
        print("pid {}, passagelen {}, shape {}".format(p_id, passage_len, a.shape))

    return p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(token_ids, np.int32).tobytes()


in_path = os.path.join(args.wiki_dir,"hotpot_wiki.tsv")
out_passage_path = os.path.join(args.out_data_dir, "passages")
print(in_path)

configObj = MSMarcoConfigDict['dpr']
tokenizer = configObj.tokenizer_class.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True,
    cache_dir=None,
)

with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f, open('{}_split{}'.format(out_passage_path, args.num), 'wb') as out_f:
    for idx, line in enumerate(in_f):
        
        #if idx < args.num * 100000 or idx > (args.num + 1) * 100000:
        #    continue
        print(idx)
        out_f.write(PassagePreprocessingFn(args, line, tokenizer))