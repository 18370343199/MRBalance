# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import traceback
from argparse import ArgumentParser
from requests import ConnectTimeout, JSONDecodeError, ReadTimeout
import openai
from openai import OpenAI
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from itertools import islice
from utils.build_ICL_data import get_ESC_demonstrations, get_esc_COT
import csv
from utils.log_kits import get_logger


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_davinci(args, logger, **kwargs):
    try:
        response = client.chat.completions.create(**kwargs)
        response_ls = [response["choices"][i]["text"] for i in range(len(kwargs["prompt"]))]
        return response_ls
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e


RETRIES = 25
MAX_TOKENS = 1024  # 最长生成长度
MODEL = "gpt-3.5-turbo-1106"
client = OpenAI(base_url="",
                api_key="sk-",)


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_chat(args, logger, **kwargs):
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e

class InputExample():
    def __init__(self, input_TXT, event1, event2, labels, type):
        self.input_TXT = input_TXT
        self.event1 = event1
        self.event2 = event2
        self.labels = labels
        self.type = type

roles = [
    "You are an incident analyst who is good at identifying and interpreting events from text. You will analyze whether there is a causal relationship between events based on the context, combining the chronological order, the actions and their effects, to give your judgment.",
    "You are a causal inference model developer who focuses on judging causal relationships between two events based on the principles of logic and causal reasoning. You will analyze the events in the text based on the inference model, consider the causes and consequences of the events and the corresponding logical relationships, and give detailed explanations.",
    "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "You are an expert in natural language processing, focusing on extracting causal cues from language. You will analyze whether there is a causal connection between two events based on the explanation of causal triggers, temporal relationships, and character actions, and make judgments based on the logical structure in the language.",
    """
You are a causal analysis expert, specializing in identifying and reasoning about causal relationships between events. You need to accurately find causally related event pairs from the provided text and explain their logical connections.
    """,
    """
You are a news event analyst, specializing in understanding the background and logic of social events to identify causal links between events reported in the news.
    """,
    "You are a Machine Learning Researcher with expertise in causal inference and artificial intelligence. Assess the causal relationship between two events using advanced algorithms and techniques such as causal discovery algorithms, reinforcement learning, and deep learning models. Implement causal graphs, counterfactual analysis, and other AI-driven methods to derive and validate your conclusions.",
    "You are a Philosopher of Science specializing in the theory of causation. Critically evaluate the causal relationship between two events by exploring different philosophical theories of causation, such as counterfactual, probabilistic, and mechanistic approaches. Discuss the conceptual frameworks and logical structures that underpin causality in the given context.",
    "You are an Engineer/System Analyst with expertise in system optimization and causal relationship identification. Examine the causal link between two events by utilizing methodologies like fault tree analysis, system dynamics modeling, and root cause analysis. Apply engineering principles and quantitative assessments to determine the causality and suggest improvements or solutions.",
    "You are a Social Scientist with expertise in sociological and political analysis. Investigate the causal relationship between two social or political events using qualitative and quantitative methods, including regression analysis, structural equation modeling, and case study comparisons. Incorporate theories and frameworks relevant to social behavior and political dynamics to support your analysis.",
    "You are a social science researcher experienced in analyzing social events and relationships. Evaluate the following sentence to identify any causal relationships between the specified events, incorporating relevant social theories in your explanation.",
    "You are Wangxuemeng, a genius girl who can grasp the cause and effect of things very well, and is good at using logical methods, causal analysis methods, and graph analysis methods to solve causal identification problems"
]

tt="""In the following, you'll be given two events. Your task is to determine if there is a causal relationship between them. Answer with (A) cause, (B) caused by, or (C) None. If the events are related causally, choose the correct option. If there is no causal relationship, select option (C). The correct causal relationship might be direct, indirect, or conditional.
"""
def process(args):
    openai.api_key = args.openai_api_key

    logger = get_logger(os.path.join(args.output_dir, "log"), is_console=True)

    logger.info(str(args))

    for key, value in args.__dict__.items():
        logger.info("%s :\t%s" % (key, str(value)))

    ICL_str = ""
    if args.ICL == "zero-shot-COT":
        ...
    elif args.ICL != "None":
        if args.ICL.startswith("COT_"):
            icl_method = get_esc_COT
            # COT_2:3
            pos_num, neg_num = [int(t) for t in args.ICL[4:].split(":")]
        else:
            # 2:3
            icl_method = get_ESC_demonstrations
            pos_num, neg_num = [int(t) for t in args.ICL.split(":")]

        ICL_list = icl_method(pos_num, neg_num)
        ICL_str = "\n\n".join(ICL_list) + "\n\n"

    with open(args.input_file, 'r', encoding="utf-8") as fin:
        with open(os.path.join(args.output_dir, "response.json"), 'w+', encoding="utf-8") as fout:
            examples = []
            reader = csv.reader(fin)
            for row in islice(reader, 1, None):
                input_TXT = re.sub(r'causal </s></s> unrelated </s></s>', '', row[0])
                event1 = row[2]
                event2 = row[3]
                labels = row[4]
                type = row[5]
                examples.append(
                    InputExample(input_TXT=input_TXT, event1=event1, event2=event2, labels=labels, type=type))
            batch_size = 2
            item_buckets = [examples[t:t + batch_size] for t in range(0, len(examples), batch_size)]

            # total ? item_buckets
            for bucket_no, item_buck in enumerate(item_buckets):
                if args.debug and bucket_no == 2:
                    break
                logger.info("#" * 10 + "bucket " + str(bucket_no + 1) + "#" * 10)

                prompts = []

                for item in item_buck:
                    sent = item.input_TXT
                    event1 =item.event1
                    event2 =item.event2

                    if item.type=="inter":
                        prompt = """
Input: %s
Question: What is the causal relationship between \"%s\" and \"%s\" ? Answer as requested, don't ask me back(answer format "Answer:(A),(B) or (C)").
(A) cause
(B) caused by
(C) None
""" % (sent, event1, event2)
                    else:
                        prompt = """
Input: %s
Question: What is the causal relationship between \"%s\" and \"%s\" ? Answer as requested, don't ask me back(answer format "Answer:(A),(B) or (C)").
(A) cause
(B) caused by
(C) None
""" % (sent, event1, event2)

                    prompts.append(prompt)

                if args.gpt_type in ("text-davinci-002", "text-davinci-003"):
                    response_ls = completion_with_backoff_davinci(args, logger,
                                                                  model=args.gpt_type,
                                                                  prompt=prompts,
                                                                  temperature=0,
                                                                  max_tokens=1024,
                                                                  )
                elif args.gpt_type in ("gpt-3.5-turbo-1106", "gpt-4"):
                    response_ls = []
                    for prompt in prompts:
                        response = completion_with_backoff_chat(args, logger,
                                                                model=args.gpt_type,
                                                                messages=[
                                                                    {"role": "system", "content": roles[4]},
                                                                    {"role": "user", "content": prompt}
                                                                ],
                                                                temperature=0,
                                                                max_tokens=1024,
                                                                )
                        response_ls.append(response.choices[0].message.content)
                else:
                    assert False

                assert len(response_ls) == len(prompts)
                # feed responses to json
                for item, response, prompt in zip(item_buck, response_ls, prompts):
                    logger.info("\nPrompt+response:\n" + prompt + response)
                    t={}
                    t["sentence"]=item.input_TXT
                    t["event1"]=item.event1
                    t["event2"]=item.event2
                    t["type"]=item.type
                    if item.labels=="NONE":t["labels"]=0
                    else: t["labels"]=1

                    t["gpt_response"] = response

                    try:
                        tmp_str = response.split("Answer:", maxsplit=1)[-1]
                    except:
                        tmp_str = response

                    gpt_pred_bi_causal_label = 0
                    tmp_words = re.split("[^a-z]", tmp_str.strip().lower())
                    for w in tmp_words:
                        if w == "a" or w=="b":
                            gpt_pred_bi_causal_label = 1
                            break
                        elif w == "c":
                            gpt_pred_bi_causal_label = 0
                            break
                    t["gpt_pred_bi_causal_label"] = gpt_pred_bi_causal_label

                    fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                    fout.flush()


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



