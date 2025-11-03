# -*- coding: utf-8 -*-
import math
import json
import os
import re
import shutil
import traceback
from argparse import ArgumentParser
from itertools import islice

from requests import ConnectTimeout, JSONDecodeError, ReadTimeout
import openai
from openai import OpenAI
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# from util import *
from tools import *
import csv
from utils.build_ICL_data import get_ESC_demonstrations, get_esc_COT
import sys
from utils.log_kits import get_logger
from LLMLP import LLMLP



def set_rd_seed(seed):
    random.seed(seed)




ROLES = ["knowledge graphs expert", "knowledge graphs expert", "knowledge graphs expert", "knowledge graphs expert"]
Prompt_tuning = ["positive", "normal", "normal", "negative"]
models = ["gpt-4o-mini-2024-07-18","gpt-4o-mini-2024-07-18","gpt-4o-mini-2024-07-18","gpt-4o-mini-2024-07-18"]


def process(args):

    logger = get_logger(os.path.join(args.output_dir, "log"), is_console=True)
    logger.info(str(args))

    for key, value in args.__dict__.items():
        logger.info("%s :\t%s" % (key, str(value)))


    set_rd_seed(0)
    # assert len(ROLES) > 0
    # assert len(JUDGES) > 0
    # os.makedirs(DIR_NAME, exist_ok=True)
    total_responses = 0
    response_dict={}
    ACTIVATION = "listwise"
    TYPE = "single_choice"
    llmlp = LLMLP(args.gpt_type, len(ROLES), ROLES, Prompt_tuning,3, ACTIVATION, TYPE, args.gpt_type,models)
    accs, resp_cnts, importances = [], 0, []
    completion_list = []
    total_prompt_tokens, total_completion_tokens = 0,0
    with open(args.input_file, 'r', encoding="utf-8") as fin:
        with open(os.path.join(args.output_dir, "response.json"), 'w+', encoding="utf-8") as fout:
            item_list = [json.loads(line) for line in fin.readlines()]
            batch_size = 1
            item_buckets = [item_list[t:t + batch_size] for t in range(0, len(item_list), batch_size)]
            # item_buckets = item_buckets[185:]
            # total ? item_buckets
            for bucket_no, item_buck in enumerate(item_buckets):
                if args.debug and bucket_no == 2:
                # if bucket_no == 2:
                    break
                # if bucket_no<309: continue
                logger.info("#" * 10 + "bucket " + str(bucket_no + 1) + "#" * 10)

                prompts = []
                # ans = []
                for item in item_buck:
                    words = item["words"]
                    sent = " ".join(words)
                    events = item["events"]
                    event1 = " ".join([words[t] for t in events[0]])
                    event2 = " ".join([words[t] for t in events[1]])
                    # ans.append(item["bi_causal_label"])
                    
                    prompt = """
Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
(A) cause
(B) caused by
(C) none""" % (sent, event1, event2)

                    prompts.append(prompt)



                response_ls = []
                for index, prompt in enumerate(prompts):
                    llmlp.zero_grad()
                    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(prompt,logger)
                    imp_score = llmlp.backward(res)

                    completion_list.append(completions)

                    resp_cnts += resp_cnt
                    importances.append(imp_score)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    print("total prompt tokens:{}, total completion tokens:{}".format(total_prompt_tokens,total_completion_tokens))
                    response_ls.append(res)
                # feed responses to json
                for item, response, prompt in zip(item_buck, response_ls, prompts):
                    logger.info("\nPrompt+response:\n" + prompt +"\n"+ str(response))
                    item["gpt_response"] = response
                    gpt_pred_bi_causal_label = 0
                    if response is None: response="C"
                    tmp_words = re.split("[^a-z]", response.strip().lower())
                    for w in tmp_words:
                        if w == "a" or w=="b":
                            gpt_pred_bi_causal_label = 1
                            break
                        elif w == "c":
                            gpt_pred_bi_causal_label = 0
                            break
                    item["gpt_pred_bi_causal_label"] = gpt_pred_bi_causal_label
                    
                    print(item)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
            # json.dump(response_dict, open(
            #         args.output_dir + "/{}_{}_{}.json".format("gpt35_1106", agents, rounds), "w"))
    print("total prompt tokens:{}, total completion tokens:{}".format(total_prompt_tokens,total_completion_tokens))


if __name__ == '__main__':
    """
    set "input_file" and "gpt_type" to control select dataset and ChatGPT's version 

    set "openai_api_key" as your own API key.

    set "ICL" as:
    1. "None": perform zero-shot ChatGPT
    2. "x:y": perform ICL with x positive and y negative demonstrations, such as "2:3".
    3. "zero-shot-COT": perform ChatGPT with zero-shot COT, i.e, add "Let's think step by step." into the prompt.
    4. "COT_x:y": perform ChatGPT with COT, with x positive and y negative demonstrations, such as "COT_4:8".

    For example:

    python ECI.py \
    --input_file data/CTB/causal_time_bank.json \
    --output_dir output/CTB \
    --gpt_type text-davinci-002 \
    --openai_api_key YOUR_API_KEY \
    --ICL COT_4:2
    """
    parser = ArgumentParser()

    parser.add_argument('--input_file',
                        default="../data/ESC_intra_data.json",
                        type=str,
                        choices=["../data/ESC_intra_data.json","../data/CTB_intra_data.json"])
    parser.add_argument('--output_dir', default="./outputs/gpt-4o-mini-2024-07-18/pnonone", type=str)
    parser.add_argument('--gpt_type', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["doubao-1-5-lite-32k-250115", "gpt-4.1-mini-2025-04-14", "gpt-4o-mini", "gpt-4o-mini-2024-07-18","gpt-4o-2024-11-20",
                                 "doubao-seed-1.6-flash", "qwen-max-2025-01-25", "qwen2.5:7b","qwen2.5:14b","qwen2.5:32b","deepseek-r1","o3-mini"])
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    process(args)



