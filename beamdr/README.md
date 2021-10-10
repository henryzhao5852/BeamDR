

# BeamDR Retrieval

## Install

python setup.py install

Note: we use Pytorch 1.2.0 and Transformers 2.4.1, we recommend using virtual environments, as the required packages are different from the reader. 


## Data Download
All the data locations in put in command/data.py

## Inference on existing checkpoint

1. download the hotpotQA [dev data](https://www.dropbox.com/s/qrr15hgnw2uta15/eval_data.tar.gz?dl=0) and [passages](https://www.dropbox.com/s/q37b3cuaom0pq5k/ori_data.tar.gz?dl=0), [dev type classification results](https://www.dropbox.com/s/5o6x54bg6js39uk/dev_type_results.pkl?dl=0), [model checkpoints](https://www.dropbox.com/s/w9y50yiqnvie148/beamdr_checkpoint.tar.gz?dl=0), [first hop query embeddings](https://www.dropbox.com/s/p9yvyish12jqtat/first_dev_query_embs.tar.gz?dl=0) and [second hop query embeddings](https://www.dropbox.com/s/9c30od4pwiznrbb/sec_dev_query_embs.tar.gz?dl=0), [first hop passage embeddings](https://www.dropbox.com/s/mx5xe06d64gkrdn/first_hop_passage.tar.gz?dl=0) and [second hop passage embeddings](https://www.dropbox.com/s/c37l0nwnvbla3vt/sec_hop_passage.tar.gz?dl=0), then put them on related directories. 
2. Run evaluation on first hop (no GPU required)
```bash
        python -m torch.distributed.launch --nproc_per_node=1 drivers/eval_test.py \
         --model_type dpr \
         --output_dir /path/to/out/data/  \
         --passage_path /path/to/wikipedia/file/ \
         --data_dir /path/to/all/data/file/ \
         --topk 30 \
         --model /path/to/model/checkpoint/checkpoint-237000 \
         --no_cuda \
         --local_rank -1 \
         --load_cache \
         --latest_num 237000 \
         --world_size 4
```
3. Run evaluation on second hop (no GPU required)
```bash

        python -m torch.distributed.launch --nproc_per_node=1 drivers/eval_sec_test.py \
         --model_type dpr \
         --output_dir /path/to/out/data/ \
         --data_dir /path/to/all/data/file/ \
         --topk 100  \
         --passage_path /path/to/wikipedia/file/ \
         --model /path/to/model/checkpoint/checkpoint-165000 \
         --latest_num 165000 \
         --world_size 4 \
         --no_cuda \
         --local_rank -1 \
         --load_cache
```
4. The .pkl file will be generated, run 
```bash

        python utils/eval_retrieval.py \
        --data_dir /path/to/all/data/file/ \
        --first_hop_file /path/to/all/first/hop/file/ \
        --sec_hop_file /path/to/all/second/hop/file/
```
We include the development set [retrieval results](https://www.dropbox.com/s/78ob16em28ayc82/dev_retrieval_results.tar.gz?dl=0) (.pkl) , and top 10 [reranking results](https://www.dropbox.com/s/dd4yx2ten03gffp/dev_rerank_top10.pkl?dl=0) here. 

## Inference 

If you want to evaluate you model from scratch, download [hotpotQA dev data](https://www.dropbox.com/s/q37b3cuaom0pq5k/ori_data.tar.gz?dl=0), [type classification results](https://www.dropbox.com/s/5o6x54bg6js39uk/dev_type_results.pkl?dl=0), [splited Wikipedia data](https://www.dropbox.com/s/88gllfnt8xmuwo7/passages.zip?dl=0) (or run data/passage_encode.py, we split on every 100K passages), then run 
Step 1:
```bash

python data/DPR_data_test.py \
--out_data_dir /path/to/out/data/ \
--question_dir /path/to/question/data \
--wiki_dir /path/to/wikipedia/file/
```

Step 2 (Note: it requires GPU for embeddings, and --load_cache flag should keep false):
```bash
         python -m torch.distributed.launch --nproc_per_node=4 drivers/eval_test.py \
         --model_type dpr \
         --output_dir /path/to/out/data/  \
         --passage_path /path/to/wikipedia/file/ \
         --data_dir /path/to/all/data/file/ \
         --topk 30 \
         --model /path/to/model/checkpoint/  \
         --latest_num checkpoint_num 
```

Step 3:
```bash

        python data/DPR_data_test_sec.py \
        --out_data_dir /path/to/out/data/ \
        --question_dir /path/to/question/data \
        --wiki_dir /path/to/wikipedia/file/
```

Step 4:
```bash


        python -m torch.distributed.launch --nproc_per_node=1 drivers/eval_sec_test.py  \
        --model_type dpr \
        --output_dir /path/to/out/data/ \
        --data_dir /path/to/all/data/file/ \
        --topk 100  --passage_path /path/to/wikipedia/file/ \
        --model /path/to/model/checkpoint/ \
        --latest_num checkpoint_num 
```

You will then get both first hop and second hop predictions.


## Training

To train the model(s) in the paper, you need to start two commands in the following order:

Step 1 Data preprocessing:

Download [training and dev data](https://www.dropbox.com/s/2szpsy49qrb257d/training_data.tar.gz?dl=0), [wiki data](https://www.dropbox.com/s/q37b3cuaom0pq5k/ori_data.tar.gz?dl=0), [splited Wikipedia data](https://www.dropbox.com/s/88gllfnt8xmuwo7/passages.zip?dl=0), and [DPR checkpoints](https://www.dropbox.com/s/rqpelr1v54ltedy/dpr_checkpoint.tar.gz?dl=0) for warmup, the run the following: 

```bash

python data/DPR_data.py \
--out_data_dir /path/to/out/data/ \
--question_dir /path/to/question/data \
--wiki_dir /path/to/wikipedia/file/
```

Step 2 Initial ANN data generation: 
this step will use the pretrained BM25 warmup checkpoint to generate the initial training data. The command is as follow: (Note: First hop and second hop training are separate, we suggest putting files on different directories. )

```bash

python -m torch.distributed.launch --nproc_per_node=4 drivers/run_ann_data_gen_dpr.py \
--training_dir /path/to/training/dir/  \
--model_type dpr \
--output_dir /path/to/output/dir/  \
--data_dir /path/to/data/dir/ \
 --topk_training 10 \
 --negative_sample 8 \
 --passage_path /path/to/psg/dir/  \
 --init_model_dir /path/to/dpr/checkpoint/
```

Second hop:

```bash

python -m torch.distributed.launch --nproc_per_node=4 drivers/run_ann_data_gen_dpr_sec_hop.py \
--training_dir /path/to/training/dir/  \
--model_type dpr \
 --output_dir /path/to/output/dir/ \
  --data_dir /path/to/data/dir/ \
  --topk_training 10 \
 --negative_sample 8 \
 --passage_path /path/to/psg/dir/ \
 --init_model_dir /path/to/dpr/checkpoint/
```

c. Training: ANCE training with the most recently generated ANN data, the command is as follow:
```bash

python -m torch.distributed.launch --nproc_per_node=4 ../drivers/run_ann_dpr.py \
--model_type dpr \
--model_name_or_path /path/to/dpr/checkpoint/   \
--data_dir /path/to/data/dir/ \
--ann_dir /path/to/ann_data/ \
--max_seq_length 256 \
--per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5 \
--output_dir /path/to/output/dir/ \
--warmup_steps 1237 \
--logging_steps 100 \
--save_steps 3000 \
--log_dir runs/ 
```
Note: Ann data generation and ANN training should run simultaneously, since the data is generated by the being optimized model.
		
## Acknowledgments

The code is adapted from [ANCE](https://github.com/microsoft/ANCE). Huge thanks to the contributors!
