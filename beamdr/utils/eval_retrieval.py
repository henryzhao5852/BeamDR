import json 
import pickle
import unicodedata
from tqdm import tqdm
import pickle 
import argparse
import os 
import csv 
def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()

parser = argparse.ArgumentParser()
parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
        "--first_hop_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
        "--sec_hop_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)



args = parser.parse_args()
passage_path = os.path.join(args.data_dir, "hotpot_wiki.tsv")
title2text = dict()
with open(passage_path, "r", encoding="utf-8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', )
    # file format: doc_id, doc_text, title
    for row in reader:
        if row[0] != 'id':
            title2text[row[2]] = row[1]

sec_hop_pairs = pickle.load(open(args.sec_hop_file, 'rb'))
print(len(sec_hop_pairs))
first_hop_ets = pickle.load(open(args.first_hop_file, 'rb'))
type_dict = pickle.load(open(args.data_dir + '/dev_type_results.pkl', 'rb'))

with open(args.data_dir + '/hotpot_dev_fullwiki_v1.json', 'r') as fin:
    dataset = json.load(fin)

total = 0
total_ans = 0
em_count = 0
p_em_count = 0 
ar_count = 0 
pr_count = 0

for i, data in tqdm(enumerate(dataset)):
    qid = data['_id']
    
    #if data['answer'].lower() in ['yes', 'no']:
    #    continue
    supp_set = set()
    for supp in data['supporting_facts']:
        title = supp[0]
        supp_set.add(normalize(title))
    total += 1
    supp_set = list(supp_set)


    if type_dict[qid] == 'comparison':
        
        pred = first_hop_ets[qid]['pred'][:20]

        if supp_set[0] in pred and supp_set[1] in pred:
            p_em_count += 1
        if supp_set[0] in pred or supp_set[1] in pred:
            pr_count += 1
        if supp_set[0] in pred[:2] and supp_set[1] in pred[:2]:
            em_count += 1

    else:
        et_pairs = sec_hop_pairs[qid]
        pred = list()
        for f_et, s_et in et_pairs[:10]:

            if f_et not in pred:
                pred.append(f_et)
            if s_et not in pred:
                pred.append(s_et)
        
        if supp_set[0] in pred and supp_set[1] in pred:
            p_em_count += 1
        if supp_set[0] in pred or supp_set[1] in pred:
            pr_count += 1
        if supp_set[0] in pred[:2] and supp_set[1] in pred[:2]:
            em_count += 1
    if data['answer'].lower() in ['yes', 'no']:
        continue
    total_ans += 1
    for title in pred:
        if data['answer'].lower() in ''.join(title2text[title]).lower():
            ar_count += 1
            break
                
print('Exact match: {}/{} = {}%'.format(em_count, total, em_count / total * 100))
print('Passage exact match: {}/{} = {}%'.format(p_em_count, total, p_em_count / total * 100))
print('Passage recall: {}/{} = {}%'.format(pr_count, total, pr_count / total * 100))
print('Answer recall: {}/{} = {}%'.format(ar_count, total_ans, ar_count / total_ans * 100))

