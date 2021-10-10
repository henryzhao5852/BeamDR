

# GRR Reader 

BERT Based Span extraction model, the repo is adapted from https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths/tree/master/reader

## Install 

We suggest creating a vitural environment, and we use pytorch 1.2.0 and pytorch_transformers 1.2.0 for this repo.

pip install -r requirements.txt


## Data Download
We provide [training data](https://www.dropbox.com/s/0rp40y5p7xxrxur/reader_train_data.json?dl=0) and [evaluation data](https://www.dropbox.com/s/x7uddrmivj1zzjn/reader_dev.json?dl=0) extracted from BeamDR (reranked) outputs. We also provide BERT base [model checkpoint](https://www.dropbox.com/s/dhj04830b52wroj/reader_bert_base.tar.gz?dl=0), [predictions](https://www.dropbox.com/s/w8hv8r0t4n0xv2v/bert_base_pred.json?dl=0) and BERT large [model checkpoint](https://www.dropbox.com/s/663y4gp8uvw62vn/reader_bert_large.tar.gz?dl=0), [predictions](https://www.dropbox.com/s/tjgjjh3o4hiy8xx/bert_large_pred.json?dl=0). 

## Training

```bash
python run_reader_confidence.py \
    --bert_model bert-base-uncased \
    --output_dir output_hotpot_bert_base \
    --train_file data/hotpot/hotpot_reader_train_data.json \
    --predict_file data/hotpot/hotpot_dev_squad_v2.0_format.json \
    --max_seq_length 384 \
    --do_train \
    --do_predict \
    --do_lower_case \
    --version_2_with_negative 
```

## Evaluation

The evaluation script is available [here](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/), 
the official data provided by GRR authors is available [here](https://drive.google.com/file/d/1MysthH2TRYoJcK_eLOueoLeYR42T-JhB/view). 

## Acknowledgments

The code is adapted from [repo](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths) from "Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering". Huge thanks to the contributors!
