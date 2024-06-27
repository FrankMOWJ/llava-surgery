#!/bin/bash
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2024-05-02 10:23:39
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2024-06-06 22:49:02
 # @FilePath: /LLaVA/scripts/v1_5/eval/textvqa.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
python -m llava.eval.model_JIGSAWS_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --model-base ./pretrain_model/models--liuhaotian--llava-v1.5-7b \
    --question-file ./playground/JIGSAWS_test/llava_textvqa_JIGSAWS.jsonl \
    --image-folder ./data \
    --answers-file ./playground/JIGSAWS_test/answers/llava_textvqa_JIGSAWS_text_ans.json \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --mode text

# python -m llava.eval.eval_textvqa \
#     --annotation-file playground/JIGSAWS_test/llava_textvqa_JIGSAWS_annatation.json \
#     --result-file ./playground/JIGSAWS_test/answers/llava_textvqa_JIGSAWS_ans.json