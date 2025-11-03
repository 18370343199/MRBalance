# -*- coding: utf-8 -*-
import math
import json
import os
import re
import shutil
import traceback
from argparse import ArgumentParser
from requests import ConnectTimeout, JSONDecodeError, ReadTimeout
import openai

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# from util import *
from tools import *

from utils.build_ICL_data import get_ESC_demonstrations, get_esc_COT
import sys
from utils.log_kits import get_logger
# PART = int(sys.argv[1])
# EXP_NAME = sys.argv[2]
# MODEL = sys.argv[3]
#
# ACTIVATION = "listwise"
# TYPE = "code_completion"
# # ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
# DIR_NAME = sys.argv[4]
# ROLES = ast.literal_eval(sys.argv[5])
# JUDGES = ast.literal_eval(sys.argv[6])
# DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)
#
# SUBSET = 50

# @retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
# def completion_with_backoff_davinci(args, logger, **kwargs):
#     try:
#         response = client.chat.completions.create(**kwargs)
#         response_ls = [response["choices"][i]["text"] for i in range(len(kwargs["prompt"]))]
#         return response_ls
#     except Exception as e:
#         logger.warning("\n\n" + traceback.format_exc() + "\n\n")
#         raise e


RETRIES = 25
MAX_TOKENS = 1024  # 最长生成长度
# BETA_URL = "http://43.163.219.59:8001/beta"
from openai import OpenAI

# add your api key
client = OpenAI(
    api_key="",
    base_url="",
)
# response = client.chat.completions.create(model="gemma2:9b",messages=[{"role": "user", "content": "who are you"}])

@retry(wait=wait_random_exponential(min=10, max=5000), stop=stop_after_attempt(10))
def completion_with_backoff_chat(args, logger,model, **kwargs):
    try:
        if model in("gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125","gpt-4o-mini"):
            response = client.chat.completions.create(
                model=model,
                **kwargs)
        else:
            response = client2.chat.completions.create(
                model=model,
                **kwargs)
        return response
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e



def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None


def parse_ABCD(input_str):
    # 匹配 'yes' 或 'no'，忽略大小写
    if input_str==None:
        input_str="Answer:(C)"
    try:
        tmp_str = input_str.split("Answer:", maxsplit=1)[-1]
    except:
        tmp_str = input_str
    solution = None

    tmp_words = re.split("[^a-z]", tmp_str.strip().lower())
    for w in tmp_words:
        if w == "a":
            solution="a"
            break
        elif w == "b":
            solution="b"
            break
        elif w=="c":
            solution = "c"
            break

    return solution
def check_reach_consensus(logger,agent_contexts):
    pred_solutions = [context[-1]["content"] for context in agent_contexts]
    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_ABCD(pred_solution)

        # if pred_answer is None:
        #     pred_answer = solve_math_problems(pred_solution)
        #     print(pred_solution)

        if pred_answer is not None:
            pred_answers.append(pred_answer)

    # filter except ABCD
    pred_answers = [answer for answer in pred_answers if answer in ["A","B","C","a","b","c"]]
    logger.info(pred_answers)
    if len(pred_answers) == 0:
        print("No answer found")
        return False,None

    def most_frequent(List):
        counter = 0
        num = List[0]
        # a=0
        # b=0
        # c=0
        for i in List:
            current_frequency = List.count(i)

            # if i=="a": a=current_frequency
            # elif i=="b": b=current_frequency
            # else: c=current_frequency

            if current_frequency > counter:
                counter = current_frequency
                num = i

        # if a+b>c:
        #     num="a"
        #     counter=a+b
        # else:
        #     num="c"
        #     counter=c

        return num, counter

    consensus_answer, counter = most_frequent(pred_answers)
    if counter > math.floor(2 / 3 * len(agent_contexts)):
        logger.info("\nConsensus answer: {}".format(consensus_answer))
        # print("Consensus answer: {}".format(consensus_answer))
        return True,consensus_answer
    return False,None

def extract_causal_answer(pred_str):
    pred = pred_str.split('Answer:')[-1].strip()
    return pred


def set_rd_seed(seed):
    random.seed(seed)

def construct_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form \"Answer: (A),(B) or (C)\" at the end of your response. "}

    prefix_string = "Here is the question: " + question + "\n\nThese are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"]
        response = "\n\nOne agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Put your final answer in the form \"Answer: (A),(B) or (C)\nRationale: Your reason\" """
    return {"role": "user", "content": prefix_string}

SYSTEM_PROMPT = "It's a debate. Explain your reasons at each round thoroughly.\nAll questions are causal reasoning questions."
# roles =[
#     "You are an expert in natural language processing, focusing on extracting causal cues from language. You will analyze whether there is a causal connection between two events based on the explanation of causal triggers, temporal relationships, and character actions, and make judgments based on the logical structure in the language.",
#     "You are a causal analysis expert, specializing in identifying and reasoning about causal relationships between events. You need to accurately find causally related event pairs from the provided text and explain their logical connections.",
#     "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
#     "You are a causal analysis expert, specializing in identifying and reasoning about causal relationships between events. You need to accurately find causally related event pairs from the provided text and explain their logical connections.",
# ]
roles =[
    "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
]
prompt_roles={
    "none":"",
    "positive":"""
Here are some rules and examples:
1. **Task Definition Prompt**:
   - "Identify the causal relationship between two events in the provided text. Choose the most appropriate answer based on the options: (A) 'cause', (B) 'caused by', (C) 'none'. Use time precedence, covariation, and logical dependency to decide."

2. **Condition Prompt**:
   - "If Event A leads to the occurrence of Event B, select (A) 'cause'. If Event B is the reason for the occurrence of Event A, choose (B) 'caused by'. If there’s no causal connection, select (C) 'none'."

3. **Sample Prompts for Each Choice**:
   - Sample 1: `input: Due to heavy rains, the river level rose significantly, causing floods in nearby areas.`  
     Question: What is the relationship between "heavy rains" and "floods in nearby areas"?  
     (A) cause  
     (B) caused by  
     (C) none  
     Answer: (A)

   - Sample 2: `input: The research team found that the increasing cases of mental stress among students were largely due to excessive screen time.`  
     Question: What is the relationship between "excessive screen time" and "mental stress"?  
     (A) cause  
     (B) caused by  
     (C) none  
     Answer: (B)

   - Sample 3: `input: The innovative approaches in renewable energy are advancing rapidly, while urbanization continues to increase in densely populated areas.`  
     Question: What is the relationship between "innovative approaches in renewable energy" and "urbanization"?  
     (A) cause  
     (B) caused by  
     (C) none  
     Answer: (C)

4. **Time-Based Causal Prompt**:
   - "If Event A happens before Event B and B’s occurrence depends on A, choose (A). If Event A depends on Event B, choose (B). Select (C) if no causal connection exists."

5. **Semantic Causal Prompt**:
   - "Assess the semantic and logical connection between Event A and Event B in the text. Consider mutual influence and the presence of covariation to determine the causal relationship."
---
""",
    "normal":"""
Refined Rules for Identifying Causal Relationships
1. Look for Causal Indicators
Keyword Identification: Pay attention to words or phrases that indicate causality, such as "because," "leads to," "causes," "results in," "so," "therefore," "thus," "due to," "as a result," etc.
Voice Analysis: Active voice often indicates causality more directly than passive voice. For example, "Smoking causes cancer" is more direct than "Cancer is caused by smoking."
2. Determine the Temporal Order of Events
Cause Before Effect: Ensure that Event A (cause) occurs before or simultaneously with Event B (effect). If Event B occurs before Event A, the relationship may be reversed or spurious.
Avoid Temporal Reversal: If Event B occurs before Event A, carefully assess whether Event B might actually be the cause of Event A.
3. Evaluate Logic and Mechanism
Rationality: Judge whether it is logically reasonable for Event A to cause Event B, based on common sense, scientific principles, or existing theories.
Mechanism Explanation: Is there a known or plausible mechanism explaining how Event A leads to Event B? For example, smoking introduces carcinogens that damage lung cells, leading to cancer.
4. Rule Out Other Possibilities
Control Confounding Factors: Consider whether there is a third factor (confounder) that affects both Event A and Event B. For example, stress might lead to both smoking and health problems.
Independence Verification: Confirm that the relationship between Event A and Event B is not entirely explained by other factors.
5. Assess the Strength of Association
Direct Association: Is the association between Event A and Event B direct rather than indirect? For example, does smoking directly cause cancer, or is it mediated by other factors?
Frequency of Association: When Event A occurs, does Event B often or always occur? A strong association increases the likelihood of causality.
6. Determine the Direction of Causality
Unidirectionality: Clarify whether Event A causes Event B or Event B causes Event A. Avoid assuming causality based solely on correlation.
Possibility of Bidirectionality: If bidirectional causality exists (e.g., poverty and poor health), evaluate each direction separately.
7. Consider Specificity
Exclusivity: Does Event A cause only Event B and not other unrelated outcomes? For example, smoking causes not only lung cancer but also heart disease.
Exceptions: If Event A causes multiple outcomes, focus on its specific relationship with Event B.
8. Observe Consistency
Repeated Verification: Is the relationship where Event A causes Event B consistent across different situations or multiple observations?
Applicability Across Contexts: Does the causal relationship hold in different environments, populations, or time periods?
9. Dose-Response Relationship (if applicable)
Quantitative Changes: Does an increase in the intensity or amount of Event A lead to an increase in the intensity or occurrence of Event B? For example, does higher exposure to smoking increase the risk of cancer?
10. Empirical Support
Empirical Evidence: Is there data, experiments, or research supporting the conclusion that Event A causes Event B? For example, randomized controlled trials or longitudinal studies.
Expert Consensus: Is the causal relationship widely recognized and supported by experts in the relevant field?
11. Use Causal Inference Methods
Randomized Controlled Trials (RCTs): If possible, use RCTs to establish causality by randomly assigning subjects to treatment and control groups.
Natural Experiments: Leverage naturally occurring events that mimic randomization (e.g., policy changes or natural disasters).
Causal Diagrams (DAGs): Use directed acyclic graphs to map out potential causal relationships and identify confounding factors.
12. Validate Robustness
Sensitivity Analysis: Test whether the causal relationship remains robust under different assumptions or methods.
Replication: Ensure that the findings can be replicated in independent studies or datasets.
13. Contextualize Findings
Practical Significance: Assess whether the causal relationship has meaningful real-world implications.
Limitations: Acknowledge any limitations in the data, methods, or assumptions used to establish causality
---
    """,
    "negative":"""
there are some examples for ECI:
Input: The teacher praised the student, and the student completed the assignment.
Question: What is the causal relationship between "the teacher praised the student" and "the student completed the assignment"?
Answer: (C)
Rationale: There is no direct evidence in the sentence to suggest that the praise caused the assignment completion, so there is no causal relationship.
Input: The heavy rain caused flooding in the city, leading to severe traffic jams.
Question: What is the causal relationship between "heavy rain" and "flooding in the city"?
(A) cause
(B) caused by
(C) No causal relationship
Answer: (A)
Rationale: Heavy rain directly led to flooding, indicating that "heavy rain" was the cause of "flooding in the city".
Input: The car accident was caused by the slippery road conditions.
Question: What is the causal relationship between "the car accident" and "the slippery road conditions"?
(A) cause
(B) caused by
(C) No causal relationship
Answer: (B)
Rationale: The slippery road conditions led to the car accident, meaning the car accident was caused by the slippery road conditions.
Input: The teacher praised the student, and the student completed the assignment.
Question: What is the causal relationship between "the teacher praised the student" and "the student completed the assignment"?
Answer: (C)
Rationale: There is no direct evidence in the sentence to suggest that the praise caused the assignment completion, so there is no causal relationship.
---
    """
}
model = ["gpt-4o-mini","gpt-4o-mini","gpt-4o-mini","gpt-4o-mini"]
Prompt_tuning = ["positive","normal","negative","normal"]

def process(args):
    agents=4
    rounds=3

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
    # llmlp = CoLLMLP(MODEL, len(ROLES), ROLES, len(JUDGES), JUDGES, 3, ACTIVATION, TYPE, MODEL)
    with open(args.input_file, 'r', encoding="utf-8") as fin:
        with open(os.path.join(args.output_dir, "response.json"), 'w+', encoding="utf-8") as fout:
            item_list = [json.loads(line) for line in fin.readlines()]
            batch_size = 1
            item_buckets = [item_list[t:t + batch_size] for t in range(0, len(item_list), batch_size)]
            # item_buckets = item_buckets[123:]
            # total ? item_buckets
            for bucket_no, item_buck in enumerate(item_buckets):
                if args.debug and bucket_no == 2:
                # if bucket_no == 2:
                    break
                # if bucket_no<309: continue
                logger.info("#" * 10 + "bucket " + str(bucket_no + 1) + "#" * 10)

                prompts = []
                ans = []
                for item in item_buck:
                    words = item["words"]
                    sent = " ".join(words)
                    events = item["events"]
                    event1 = " ".join([words[t] for t in events[0]])
                    event2 = " ".join([words[t] for t in events[1]])
                    ans.append(item["bi_causal_label"])
                    prompt = """
Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?  your output should like 'Answer: (A), (B) or (C)\n'.
(A) cause
(B) caused by
(C) none""" % (sent, event1, event2)
                    

                    prompts.append(prompt)



                response_ls = []
                for index, prompt in enumerate(prompts):
                    agent_contexts = [
                        [{"role": "system", "content": roles[_]+SYSTEM_PROMPT}, {"role": "user", "content": prompt_roles[Prompt_tuning[_]]+prompt}] for _ in range(agents)
                    ]
                    store_contexts = [[{"role": "system", "content": SYSTEM_PROMPT}] for _ in range(agents)]
                    # for agent in agent_contexts:
                    #     print(agent)

                    ######### Here set your LLM model based Agent ##########

                    consensus = False
                    for i, agent_context in enumerate(agent_contexts):
                        # print(item_buck[index]["item_id"], 0, i, agent_context, "\n")
                        logger.info("{},{},{},{}\n".format(item_buck[index]["item_id"], 0, i, agent_context))
                        completion = completion_with_backoff_chat(args, logger,
                                                                model=model[i],
                                                                messages=agent_context,
                                                                temperature=0.8,
                                                                top_p=0.6,
                                                                max_tokens=1024,
                                                                )
                        logger.info(completion.choices[0].message.content)
                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)
                        store_contexts[i].extend(agent_context[1:])
                        print(completion.choices[0].message.content, "\n")
                        total_responses += 1
                        con, consens = check_reach_consensus(logger,agent_contexts[:i + 1])
                        if i >= math.floor(2 / 3 * len(agent_contexts)) and con:
                            response_dict[item_buck[index]["item_id"]] = (store_contexts[:i + 1], ans[index])
                            response_ls.append("Answer: {}".format(consens))
                            consensus = True
                            break

                    if consensus:
                        continue

                    response_dict[item_buck[index]["item_id"]] = (store_contexts, ans[index])

                    responses, gt= response_dict[item_buck[index]["item_id"]]
                    pred_solutions = []
                    max_len = max([len(response) for response in responses])
                    print("\n\n检验开始")
                    for response in responses:
                        if len(response) < max_len:
                            continue
                        pred_solution = response[-1]['content']
                        print(pred_solution)
                        pred_solutions.append(pred_solution)

                    # exit()
                    pred_answers = []
                    for pred_solution in pred_solutions:
                        pred_answer =extract_causal_answer(pred_solution)
                        pred_answers.append(pred_answer)
                    pred_answer =most_frequent(pred_answers)
                    pred_answer ="Answer: "+ pred_answer
                    print("predsanswer: "+pred_answer)

                    response_ls.append(pred_answer)
                    for res in response_ls:
                        print(res)
                assert len(response_ls) == len(prompts)
                # feed responses to json
                for item, response, prompt in zip(item_buck, response_ls, prompts):
                    logger.info("\nPrompt+response:\n" + prompt + response)
                    item["gpt_response"] = response

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
                    item["gpt_pred_bi_causal_label"] = gpt_pred_bi_causal_label
                    print(item)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
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



