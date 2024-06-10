#!/bin/sh
# Make sure you are using fschat==0.2.20 (pip install fschat==0.2.20) and transformers==4.41.2
# python3 -m fastchat.serve.model_worker --model-path /home/mossr/vicuna/vicuna-7b-v1.5
python3 -m fastchat.serve.model_worker --model-path /home/mossr/fastchat/fastchat-t5-3b-v1.0
