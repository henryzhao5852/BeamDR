import gzip
import os
import pathlib



RESOURCES_MAP = {
    'hotpotQA original data': {
        's3_url': 'https://www.dropbox.com/s/q37b3cuaom0pq5k/ori_data.tar.gz?dl=0',
        'compressed': True,
        'desc': 'Include training set, dev set and passage.tsv'
    },
    'BeamDR training data': {
        's3_url': 'https://www.dropbox.com/s/2szpsy49qrb257d/training_data.tar.gz?dl=0',
        'compressed': True,
        'desc': 'Data for training beamdr, including both hops'
    },
    'Eval data': {
        's3_url': 'https://www.dropbox.com/s/qrr15hgnw2uta15/eval_data.tar.gz?dl=0',
        'compressed': True,
        'desc': 'binary data for evaluation (include both hops in dev set)'
    },
    'Dev first hop query embeddings': {
        's3_url': 'https://www.dropbox.com/s/p9yvyish12jqtat/first_dev_query_embs.tar.gz?dl=0',
        'compressed': True,
        'desc': 'generated query embeddings for dev set first hop'
    },
    'Dev first hop passage embeddings': {
        's3_url': 'https://www.dropbox.com/s/mx5xe06d64gkrdn/first_hop_passage.tar.gz?dl=0',
        'compressed': True,
        'desc': 'generated passage embeddings for dev set first hop'
    },
    'Dev second hop query embeddings': {
        's3_url': 'https://www.dropbox.com/s/9c30od4pwiznrbb/sec_dev_query_embs.tar.gz?dl=0',
        'compressed': True,
        'desc': 'generated query embeddings for dev set second hop'
    },
    'Dev second hop passage embeddings': {
        's3_url': 'https://www.dropbox.com/s/c37l0nwnvbla3vt/sec_hop_passage.tar.gz?dl=0',
        'compressed': True,
        'desc': 'generated passage embeddings for dev set second hop'
    },
    'Splited passages': {
        's3_url': 'https://www.dropbox.com/s/88gllfnt8xmuwo7/passages.zip?dl=0',
        'compressed': True,
        'desc': 'tokenized passages, stored in binary format'
    },
    'Dev type classification': {
        's3_url': 'https://www.dropbox.com/s/5o6x54bg6js39uk/dev_type_results.pkl?dl=0',
        'compressed': True,
        'desc': 'classified dev set question type results'
    },
    'Test type classification': {
        's3_url': 'https://www.dropbox.com/s/b4qbnesqui7s9ep/test_type_results.pkl?dl=0',
        'compressed': True,
        'desc': 'classified test set question type results'
    },
    'BeamDR checkpoint': {
        's3_url': 'https://www.dropbox.com/s/w9y50yiqnvie148/beamdr_checkpoint.tar.gz?dl=0',
        'compressed': True,
        'desc': 'beamDR checkpoints, including both hops'
    },
    'DPR checkpoint': {
        's3_url': 'https://www.dropbox.com/s/rqpelr1v54ltedy/dpr_checkpoint.tar.gz?dl=0',
        'compressed': True,
        'desc': 'DPR checkpoints, including both hops'
    },
    'Dev retrieval results': {
        's3_url': 'https://www.dropbox.com/s/78ob16em28ayc82/dev_retrieval_results.tar.gz?dl=0',
        'compressed': True,
        'desc': 'Full retrieval results from Beamdr'
    },
    'Dev reranking results': {
        's3_url': 'https://www.dropbox.com/s/dd4yx2ten03gffp/dev_rerank_top10.pkl?dl=0',
        'compressed': True,
        'desc': 'Reranking results'
    },

    
}

