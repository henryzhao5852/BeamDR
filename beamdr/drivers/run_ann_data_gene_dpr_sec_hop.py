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
from torch import nn
logger = logging.getLogger(__name__)
import faiss
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import time

def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    files = list(next(os.walk(args.training_dir))[2])

    def valid_checkpoint(checkpoint):
        return checkpoint.startswith("checkpoint-")

    logger.info("checkpoint files")
    logger.info(files)
    checkpoint_nums = [get_checkpoint_no(s) for s in files if valid_checkpoint(s)]
    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" + str(max(checkpoint_nums))), max(checkpoint_nums)
    return args.init_model_dir, 0


def load_data(args):
    train_ann_path = os.path.join(args.data_dir, "train-sec-ann")
    dev_ann_path = os.path.join(args.data_dir, "dev-sec-ann")
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")

    passage_text = {}
    train_pos_id = []
    train_answers = []
    test_answers = []
    test_pos_id = []
    test_answers_trivia = []

    logger.info("Loading train ann")
    with open(train_ann_path, 'r', encoding='utf8') as f:
        # file format: q_id, positive_pid, answers
        tsvreader = csv.reader(f, delimiter="\t")
        for row in tsvreader:
            train_pos_id.append(row[1])
    
    logger.info("Loading dev ann")
    with open(dev_ann_path, 'r', encoding='utf8') as f:
        # file format: q_id, positive_pid, answers
        tsvreader = csv.reader(f, delimiter="\t")
        for row in tsvreader:
            test_pos_id.append(row[1])


    logger.info("Finished loading data, pos_id length %d, train answers length %d, test answers length %d", len(train_pos_id), len(train_answers), len(test_answers))

    return (train_pos_id, test_pos_id)


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


def generate_new_ann(args, output_num, checkpoint_path, preloaded_data, latest_step_num):
    #passage_text, train_pos_id, train_answers, test_answers, test_pos_id = preloaded_data
    #print(test_pos_id.shape)
    model = load_model(args, checkpoint_path)
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")
    
    logger.info("***** inference of train query *****")
    train_query_collection_path = os.path.join(args.data_dir, "train-sec-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)
    with train_query_cache as emb:
        query_embedding, query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=True), "query_sec_" + str(latest_step_num)+"_", emb, is_query_inference = True)
    
    logger.info("***** inference of dev query *****")
    dev_query_collection_path = os.path.join(args.data_dir, "dev-sec-query")
    dev_query_cache = EmbeddingCache(dev_query_collection_path)
    with dev_query_cache as emb:
        dev_query_embedding, dev_query_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=True), "dev_sec_query_"+ str(latest_step_num)+"_", emb, is_query_inference = True)

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, GetProcessingFn(args, query=False), "passage_"+ str(latest_step_num)+"_", emb, is_query_inference = False, load_cache = False)
    logger.info("***** Done passage inference *****")
    
    if is_first_worker():
        train_pos_id, test_pos_id = preloaded_data
        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        top_k = args.topk_training 
        #num_q = passage_embedding.shape[0]

        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(passage_embedding)
        
        logger.info('Data indexing completed.')
        
        logger.info("Start searching for query embedding with length %d", len(query_embedding))
       
        II = list()
        for i in range(15):
            _, idx = cpu_index.search(query_embedding[i * 5000 : (i + 1) * 5000], top_k) #I: [number of queries, topk]
            II.append(idx)
            logger.info("Split done %d", i)
            
        I = II[0]
        for i in range(1, 15):
            I = np.concatenate((I, II[i]), axis=0)
            



        logger.info("***** GenerateNegativePassaageID *****")
        effective_q_id = set(query_embedding2id.flatten())




        #logger.info("Effective qid length %d, search result length %d", len(effective_q_id), I.shape[0])
        query_negative_passage = dict()
        for query_idx in range(I.shape[0]): 
            query_id = query_embedding2id[query_idx]
            doc_ids = list()
                

            doc_ids = [passage_embedding2id[pidx] for pidx in I[query_idx]]


            neg_docs = list()
            for doc_id in doc_ids:
                pos_id = [int(p_id) for p_id in train_pos_id[query_id].split(',')]

                if doc_id in pos_id:
                    continue
                if doc_id in neg_docs:
                    continue
                neg_docs.append(doc_id)
            
            query_negative_passage[query_id] = neg_docs



        logger.info("Done generating negative passages, output length %d", len(query_negative_passage))

        logger.info("***** Construct ANN Triplet *****")
        train_data_output_path = os.path.join(args.output_dir, "ann_training_data_" + str(output_num))

        with open(train_data_output_path, 'w') as f:
            query_range = list(range(I.shape[0]))
            random.shuffle(query_range)
            for query_idx in query_range: 
                query_id = query_embedding2id[query_idx]
                # if not query_id in train_pos_id:
                #     continue
                pos_pid = train_pos_id[query_id]
                f.write("{}\t{}\t{}\n".format(query_id, pos_pid, ','.join(str(neg_pid) for neg_pid in query_negative_passage[query_id])))
        
        _, dev_I = cpu_index.search(dev_query_embedding, 10) #I: [number of queries, topk]


        top_k_hits = validate(test_pos_id, dev_I, dev_query_embedding2id, passage_embedding2id)




def validate(test_pos_id, closest_docs, query_embedding2id, passage_embedding2id):

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    logger.info('Matching answers in top docs...')
    scores = []
    total = 0 
    count = 0 
    for query_idx in range(closest_docs.shape[0]): 
        query_id = query_embedding2id[query_idx]
        doc_ids = [passage_embedding2id[pidx] for pidx in closest_docs[query_idx]]
        doc_ids = list(set(doc_ids))
        hits = []
        pos_id = [int(p_id) for p_id in test_pos_id[query_id].split(',')]
        total += len(pos_id)
        for i, doc_id in enumerate(doc_ids):
            if doc_id in pos_id:
                count += 1
        scores.append(hits)

    logger.info('Per question validation results len=%d', len(scores))

    top_k_hits = count / total
    logger.info('Validation results: top k documents hits accuracy %f', top_k_hits)
    return top_k_hits
    

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
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )
    parser.add_argument(
        "--init_model_dir",
        default='',
        type=str,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )
    parser.add_argument(
        "--last_checkpoint_dir",
        default='',
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
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
        "--cache_dir",
        default=None,
        type=str,
        required=False,
        help="The directory where cached data will be written",
    )
    parser.add_argument(
        "--end_output_num",
        default=-1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_doc_character",
        default= 10000, 
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=192,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default= 5, # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default= 50,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default= 5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--passage_path",
        default=None,
        type=str,
        required=True,
        help="passage_path",
    )


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

def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir)
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)
    preloaded_data = None

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        #if not os.path.exists(args.cache_dir):
        #    os.makedirs(args.cache_dir)
        preloaded_data = load_data(args)

    while args.end_output_num == -1 or output_num <= args.end_output_num:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)

        if args.only_keep_latest_embedding_file:
            latest_step_num = 0

        if next_checkpoint == last_checkpoint:
            time.sleep(60)
        else:
            logger.info("start generate ann data number %d", output_num)
            logger.info("next checkpoint at " + next_checkpoint)
            generate_new_ann(args, output_num, next_checkpoint, preloaded_data, latest_step_num)
            logger.info("finished generating ann data number %d", output_num)
            output_num += 1
            last_checkpoint = next_checkpoint
        if args.local_rank != -1:
            dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()