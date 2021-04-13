import argparse
import sys
sys.path += ['../']
import json
import logging
import os
from os.path import isfile, join
import random
import time
import csv
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
from model.models import MSMarcoConfigDict
from utils.util import (
    StreamingDataset, 
    EmbeddingCache, 
    get_checkpoint_no, 
    get_latest_ann_data,
    barrier_array_merge,
    is_first_worker,
)
from data.DPR_data import GetProcessingFn, load_mapping
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj, SimpleTokenizer, has_answer, DenseHNSWFlatIndexer
import random 
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer
)
import pickle 
from torch import nn
logger = logging.getLogger(__name__)
import faiss
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import time

import unicodedata

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path

    model = configObj.model_class(args)

    saved_state = load_states_from_checkpoint(checkpoint_path)
    model_to_load = get_model_obj(model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)

    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    return model


def InferenceEmbeddingFromStreamDataLoader(args, model, train_dataloader, is_query_inference = True, prefix =""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for batch in tqdm(train_dataloader, desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0, leave=True):
        
        idxs = batch[3].detach().numpy() #[#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence 
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:,chunk_no,:])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)


    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference

def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference = True, load_cache=False):
    inference_batch_size = args.per_gpu_eval_batch_size #* max(1, args.n_gpu)
    #inference_dataloader = StreamingDataLoader(f, fn, batch_size=inference_batch_size, num_workers=1)
    #print(prefix)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset, batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier() # directory created

    if load_cache:
        _embedding = None
        _embedding2id = None
    else:
        _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(args, model, inference_dataloader, is_query_inference = is_query_inference, prefix = prefix)

    # preserve to memory
    full_embedding = barrier_array_merge(args, _embedding, prefix = prefix + "_emb_p_", load_cache = load_cache, only_load_in_master = True) 
    full_embedding2id = barrier_array_merge(args, _embedding2id, prefix = prefix + "_embid_p_", load_cache = load_cache, only_load_in_master = True)

    return full_embedding, full_embedding2id



def generate_new_ann(args):
    #print(test_pos_id.shape)
    #model = None
    model = load_model(args, args.model)
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")
    
    latest_step_num = args.latest_num
    args.world_size = args.world_size


    logger.info("***** inference of dev query *****")
    dev_query_collection_path = os.path.join(args.data_dir, "dev-eval-query")
    dev_query_cache = EmbeddingCache(dev_query_collection_path)
    with dev_query_cache as emb:
        dev_query_embedding, dev_query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=True), "dev-eval-query_"+ str(latest_step_num)+"_", emb, is_query_inference = True, load_cache = args.load_cache)

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=False), "passage_"+ str(latest_step_num)+"_", emb, is_query_inference = False, load_cache = args.load_cache)



    dim = passage_embedding.shape[1]
    #print(dev_query_embedding.shape)
    print('passage embedding shape: ' + str(passage_embedding.shape))
    print('dev embedding shape: ' + str(dev_query_embedding.shape))

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)
    
    logger.info('Data indexing completed.')
    
    II = list()
    sscores = list()
    for i in range(2):
        score, idx = cpu_index.search(dev_query_embedding[i * 5000 : (i + 1) * 5000], args.topk) #I: [number of queries, topk]
        II.append(idx)
        sscores.append(score)
        logger.info("Split done %d", i)
        
    dev_I = II[0]
    scores = sscores[0]
    for i in range(1, 2):
        dev_I = np.concatenate((dev_I, II[i]), axis=0)
        scores = np.concatenate((scores, sscores[i]), axis=0)

    


    validate(args, dev_I, scores, dev_query_embedding2id, passage_embedding2id)





def validate(args, closest_docs, dev_scores, query_embedding2id, passage_embedding2id):

    logger.info('Matching answers in top docs...')
    scores = dict()
    
    count = 0
    total = 0
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")
    passage_path = os.path.join(args.passage_path, "hotpot_wiki.tsv")
    idx2title = dict()
    title2text = dict()
    with open(passage_path, "r", encoding="utf-8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: doc_id, doc_text, title
        for row in reader:
            if row[0] != 'id':
                idx2title[int(row[0])] = row[2]
                title2text[row[2]] = row[1]
    with open(args.data_dir + '/hotpot_dev_fullwiki_v1.json', 'r') as fin:
        dataset = json.load(fin)
    
    type_dict = pickle.load(open(args.data_dir + '/dev_type_results.pkl', 'rb'))
    instances = list()

    first_hop_ets = dict()
    for query_idx in range(closest_docs.shape[0]): 
        query_id = query_embedding2id[query_idx]
        all_scores = list()
        doc_ids = list()
        all_pred = closest_docs[query_idx]
        scs = dev_scores[query_idx]
        for i in range(len(dev_scores[query_idx])):
            if int(passage_embedding2id[all_pred[i]]) in offset2pid:
                doc_ids.append(offset2pid[int(passage_embedding2id[all_pred[i]])])
                all_scores.append(float(scs[i]))
        data = dataset[query_id]
        qid = data['_id']
        supp_set = set()
    
        for supp in data['supporting_facts']:
            title = supp[0]
            supp_set.add(normalize(title))
        
        total += len(supp_set)
        for ii, d_id in enumerate(doc_ids[:10]):
            title = normalize(idx2title[d_id])
            if title in supp_set:
                count += 1
        first_hop_ets[qid] = {'score': all_scores, 'pred': [normalize(idx2title[idx]) for idx in doc_ids]}

        if type_dict[qid] == 'comparison':
            continue
    
        for et in [normalize(idx2title[idx]) for idx in doc_ids]:
            pre_evidence = ''.join(title2text[et])
            qq = data['question'] + ' ' + '[SEP]' + ' ' + et.replace('_', ' ')  + ' ' + '[SEP]' + ' ' + pre_evidence
            instances.append({'dataset':'hotpot_dev_sec', 'question': qq, 'qid': qid, 
            'answers': list(), 'first_hop_cts': [et]})


    with open(args.data_dir + '/dev_sec_hop_data.json', 'w', encoding='utf-8') as f:
        json.dump(instances, f, indent=2)
    pickle.dump(first_hop_ets, open(args.data_dir + '/dev_first_hop_pred.pkl', 'wb'))
    logger.info("first hop coverage %f", count/total)
    #print(count, total, count/total)


    

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--passage_path",
        default='',
        type=str,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )


    parser.add_argument(
        "--model",
        default='',
        type=str,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=192,
        type=int,
        help="The starting output file number",
    )
    parser.add_argument(
        "--latest_num",
        default= 0,
        type=int,
    )
    parser.add_argument(
        "--topk",
        default= 50,
        type=int,
        help="top k from which negative samples are collected",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--load_cache", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--world_size", type=int, default=4)



    args = parser.parse_args()

    return args


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )



def main():
    args = get_arguments()
    set_env(args)
    generate_new_ann(args)


if __name__ == "__main__":
    main()