# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser

from utils.other_kits import compute_f1
"""
python ECI_compute_score.py \
--output_dir <output_dir output by ECI.py>
"""

parser = ArgumentParser()
parser.add_argument('--output_dir',default="D:\code\ChatGPT4CausalReasoning-main\\another_pro/output/ESC/multi_role/talking_p_no1_no2_ne1", type=str)
# parser.add_argument('--output_dir',default="D:\code\ChatGPT4CausalReasoning-main\\another_pro\output/ECI/ChatGPT3.5/esc2", type=str)
# parser.add_argument('--output_dir',default="D:\code\ChatGPT4CausalReasoning-main\\another_pro\output\ESC\inter\\test1", type=str)
args = parser.parse_args()
file_name = args.output_dir
file_name += "/response.json"
with open(file_name, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()
preds, golds = [], []

items = [json.loads(line) for line in lines]
# items = {str(t["words"]) + str(t["events"]): t for t in items}
# items = list(items.values())
inter_pres,inter_golds = [],[]
intra_pres,intra_golds = [],[]
for itemid, item in enumerate(items):
    gpt_pred_bi_causal_label = item["gpt_pred_bi_causal_label"]
    bi_causal_label = item["labels"]
    if item["type"] == "inter":
        inter_pres.append(gpt_pred_bi_causal_label)
        inter_golds.append(bi_causal_label)
    else:
        intra_pres.append(gpt_pred_bi_causal_label)
        intra_golds.append(bi_causal_label)
    preds.append(gpt_pred_bi_causal_label)
    golds.append(bi_causal_label)
###  all
p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(golds, preds, details="full", target_label_id=1)
F1_datas = [100 * t for t in (p, r, f)]

pos_acc = r

p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(golds, preds, details="full", target_label_id=0)

neg_acc = r

assert len(golds) == len(preds)
correct = 0
t=0
s=0
for g, p in zip(golds, preds):
    if g ==1:s+=1
    if g == p:
        correct += 1
        if g==1:t+=1
print(t,s)
all_acc = correct / len(golds)

print(file_name)
print("All test")
print("P,R,F= & %.1f & %.1f & %.1f" % (F1_datas[0], F1_datas[1], F1_datas[2]))
print("Pos:Neg:Full= & %.1f & %.1f & %.1f" % (pos_acc * 100, neg_acc * 100, all_acc * 100))

# inter
p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(inter_golds, inter_pres, details="full", target_label_id=1)
F1_datas = [100 * t for t in (p, r, f)]

pos_acc = r

p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(inter_golds, inter_pres, details="full", target_label_id=0)

neg_acc = r

assert len(golds) == len(preds)
correct = 0
for g, p in zip(inter_golds, inter_pres):
    if g == p:
        correct += 1

all_acc = correct / len(inter_golds)
print("inter test")
print("P,R,F= & %.1f & %.1f & %.1f" % (F1_datas[0], F1_datas[1], F1_datas[2]))
print("Pos:Neg:Full= & %.1f & %.1f & %.1f" % (pos_acc * 100, neg_acc * 100, all_acc * 100))


# intra
p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(intra_golds, intra_pres, details="full", target_label_id=1)
F1_datas = [100 * t for t in (p, r, f)]

pos_acc = r

p, r, f, c_correct, c_predict, c_gold, false_pos, false_neg, all_pos, all_neg = compute_f1(intra_golds, intra_pres, details="full", target_label_id=0)

neg_acc = r

assert len(golds) == len(preds)
correct = 0
for g, p in zip(intra_golds, intra_pres):
    if g == p:
        correct += 1

all_acc = correct / len(intra_golds)

print("intra test")
print("P,R,F= & %.1f & %.1f & %.1f" % (F1_datas[0], F1_datas[1], F1_datas[2]))
print("Pos:Neg:Full= & %.1f & %.1f & %.1f" % (pos_acc * 100, neg_acc * 100, all_acc * 100))