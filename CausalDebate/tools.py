import random
import re


def construct_ranking_message(responses, question, qtype="single_choice"):
    if qtype == "single_choice":
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        return {"role": "user", "content": prefix_string}

    elif qtype == "code_completion":
        prefix_string = "Here are some implementations of the same function and the thoughts of them. The function has a signature and a docstring explaining its functionality.\n\n"
        for aid, agent_response in enumerate(responses, start=1):
            response = "[function impl {}]:\n```method\n{}\n```\n\n".format(aid, agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + """[function signature]:\n```python\n{}\n```\n\nTake correctness, efficiency, and possible corner cases into consideration, choose top 2 solutions that match the function's docstring best. Think it step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response.""".format(question)
        return [{"role": "system", "content": prefix_string},{"role": "user", "content": prefix_string}]
    else:
        raise NotImplementedError

def parse_ranks(completion, max_num=4):
    if not isinstance(completion, str):
        content = completion.choices[0].message.content
    else:
        content = completion
    pattern = r'\[([1234]),\s*([1234])\]'
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
# def most_frequent(List):
#     counter = 0
#     num = List[0]
#
#     for i in List:
#         current_frequency = sum(is_equiv(i, item) for item in List)
#         if current_frequency > counter:
#             counter = current_frequency
#             num = i
#
#     return num
def parse_yes_no(input_str):
    # 匹配 'yes' 或 'no'，忽略大小写
    try:
        tmp_str = input_str.split("Answer:", maxsplit=1)[-1]
    except:
        tmp_str = input_str
    solution = None

    tmp_words = re.split("[^a-z]", tmp_str.strip().lower())
    for w in tmp_words:
        if w == "a":
            solution = "a"
            break
        elif w == "b":
            solution = "b"
            break
        elif w == "c":
            solution = "c"
            break

    return solution

def is_equiv(item1,item2):
    ti = parse_yes_no(item1)
    t2 = parse_yes_no(item2)
    if ti==t2:return 1
    return 0

def most_frequent(list):
    counter = 0
    num = list[0]
    print(num)
    for i in list:
        current_frequency = sum(is_equiv(i, item) for item in list)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num
