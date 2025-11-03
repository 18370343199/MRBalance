import random
import re
from util import parse_single_choice, generate_answer
# from prompt_lib import ROLE_MAP, construct_ranking_message, construct_message, SYSTEM_PROMPT_MMLU, ROLE_MAP_MATH, SYSTEM_PROMPT_MATH
from prompt_roles import roles,construct_message,system_prompt,construct_ranking_message

class LLMNeuron:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_single_choice, qtype="single_choice",prompt_tuning="positive"):
        self.role = role
        self.prompt_tuning=prompt_tuning
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None
        self.model = mtype
        # if mtype == "gpt-3.5-turbo":
        #     self.model = "gpt-3.5-turbo-1106"
        # else:
        #     raise NotImplementedError("Error init model type")

        def find_array(text):
            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                def convert(x):
                    try:
                        return int(x)
                    except:
                        return 0
                # Convert the string to a list of integers
                try:
                    ret = list(map(convert, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def activate_1(self, question, logger, bias):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        contexts.append(construct_message(formers, question, bias))

        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        logger.info(self.reply)
        if self.reply is None:
            self.reply = random.choice(['A', 'B', 'C'])
        print(self.get_reply())
        # parse answer
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            print("miss match!")
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        print([self.from_edges[eid].weight for _, eid in formers])
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])

    def activate(self, question,logger,type="intra"):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        contexts.append(construct_message(formers, question,self.prompt_tuning))
        
        self.reply, self.prompt_tokens, self.completion_tokens = generate_answer(contexts, self.model)
        logger.info(self.reply)
        if self.reply is None:
            self.reply=random.choice(['A', 'B', 'C'])
        print(self.get_reply())
        # parse answer
        self.answer = self.ans_parser(self.reply)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers):
            print("miss match!")
            weights = [0 for _ in range(len(formers))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        print([self.from_edges[eid].weight for _, eid in formers])
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])
        
    def get_context(self):
        sys_prompt = roles[self.role]
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1.reply, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question,self.prompt_tokens))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0

def parse_ranks(completion, max_num=4):
    content = completion
    pattern = r'\[([1234567]),\s*([1234567])\]'
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0])-1, int(match[1])-1]
        def clip(x):
            if x < 0:
                return 0
            if x > max_num-1:
                return max_num-1
            return x
        tops = [clip(x) for x in tops]
    except:
        print("error in parsing ranks")
        tops = random.sample(list(range(max_num)), 2)

    return tops

def listwise_ranker_2(responses, question, qtype, model="gpt-3.5-turbo"):
    # if model == "gpt-3.5-turbo":
    #     model = "gpt-3.5-turbo-1106"
    # elif model == "gpt-4":
    #     model = "gpt4"
    # else:
    #     raise NotImplementedError("Error init model type")
    assert 2 < len(responses)# <= 4
    message = construct_ranking_message(responses, question)
    completion, prompt_tokens, completion_tokens = generate_answer([message], model)
    return parse_ranks(completion, max_num=len(responses)), prompt_tokens, completion_tokens
