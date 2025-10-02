import pickle
import re
import unicodedata
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import torch



with open('all_u/backtracking_u_AWQ_Llama_8B_new.pkl', 'rb') as f:
    u_dict = pickle.load(f)

all_d = ['AIME_2024', 'FOLIO', 'temporal_sequences', 'MuSiQue100']


dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
total_input_all, total_input = [], []
for d in dataset:
    total_input.append(d['Problem'] + "\n<think>\n")
total_input_all.append(total_input)


dataset = load_dataset("yale-nlp/FOLIO")["validation"]
total_input = []
for d in dataset:
    total_input.append("Use logical deductions to determine whether the provided conclusion is true, false, or uncertain based on premise. Consider all relevant information to reach a logical conclusion.\n\n*******\nPremise: " +  d['premises'] + "\n*******\n\n*******\nConclusion: " + d['conclusion'] + "\n*******<think>\n")
total_input_all.append(total_input)


json_file = "temporal_sequences.json"
with open(json_file, "r") as f:
    dataset = json.load(f)
total_input = []
for d in dataset['examples']:
    total_input.append("Use the timeline provided and answer step by step. Finally give the index (the letter) of the actual correct answer.\n" + d['input'] + "\n<think>\n")
total_input_all.append(total_input)



with open("random_100_musique.json", 'r') as f:
    dataset = json.load(f)
total_input = []
for d in dataset:
    total_input.append(d['question'] + " Final answer of this question should be in as fewer number of words as possible." + "\n<think>\n")
total_input_all.append(total_input)


model = AutoModelForCausalLM.from_pretrained("compressed_R1_Llama_8B", device_map="auto")
model.eval()

acts = {}
def make_hook(name):
    def hook(module, input, output):
        # store the activation *with* grad
        acts[name] = output
        output.retain_grad()
        return None
    return hook

handles = []
for layer_idx, layer in enumerate(model.model.layers):
    # self_attn projections
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        module = getattr(layer.self_attn, proj)
        name   = f"layer_{layer_idx}_{proj}"
        handles.append(module.register_forward_hook(make_hook(name)))
    # mlp projections
    for proj in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(layer.mlp, proj)
        name   = f"layer_{layer_idx}_{proj}"
        handles.append(module.register_forward_hook(make_hook(name)))
        

count, total = 0, {}
all_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for layer in range(32):
    for m in all_modules:
        l_m = "layer_" + str(layer) + "_" + m
        total[l_m] = 0


##### main loop #######
for z in range(len(all_d)):
    for j in range(30):
        print(j)
        with open("output/AWQ_R1_Distill_Llama_8B/" + all_d[z] + "/" + str(j) + ".txt", "r", encoding="utf-8") as f:
            original = total_input_all[z][j] + f.read()

        with open("output/AWQ_R1_Distill_Llama_8B_output_annotated/" + all_d[z] + "/" + str(j) + ".txt", "r", encoding="utf-8") as f:
            annotated = f.read()


        labels = [
            "initializing", "deduction", "adding-knowledge",
            "example-testing", "uncertainty-estimation", "backtracking",
        ]
        seg_re = re.compile(r'\["(' + "|".join(labels) + r')"\]\s*(.*?)\s*\["end-section"\]', re.DOTALL)
        segments = [{"label": m.group(1), "text": m.group(2).strip()} for m in seg_re.finditer(annotated)]

        tokenizer = AutoTokenizer.from_pretrained("HF_models/DeepSeek-R1-Distill-Llama-8B", use_fast=True)
        enc = tokenizer(
            original,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = enc.offset_mapping
        threshold = 0.90

        flag, target_indices = False, []
        for seg in segments:
            raw = seg["text"]
            sm  = difflib.SequenceMatcher(None, original, raw)
            # find the longest matching contiguous block
            match = max(sm.get_matching_blocks(), key=lambda b: b.size)

            # how much of the snippet did we cover?
            coverage = match.size / len(raw)
            if coverage < threshold:
                continue
                raise ValueError(
                    f"Snippet {seg['label']!r} only {coverage:.0%} matched (< {threshold:.0%} required):\n{raw!r}"
                )

            # character span in the ORIGINAL string
            start_char = match.a
            end_char   = start_char + match.size

            # map to token indices
            start_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s <= start_char < e
            )
            end_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s < end_char <= e
            )

            if seg['label'] == 'backtracking':
                flag = True
                target_indices.append([start_tok, end_tok])
        if flag:
            for indices in target_indices:
                count += 1
                tokenized_input = tokenizer(original, return_tensors="pt", return_offsets_mapping=True)
                input_ids = tokenized_input["input_ids"][:, :indices[1]].to(model.device)
                attention_mask = tokenized_input["attention_mask"][:, :indices[1]].to(model.device)

                labels = input_ids.clone()
                labels[0, :indices[0]] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                model.zero_grad()
                loss.backward()

                for name, act in acts.items():
                    grad = torch.squeeze(acts[name].grad)[indices[0]:indices[1], :]
                    grad_mean = torch.mean(grad, 0)
                    total[name] += torch.abs(torch.dot(u_dict[name].to(grad_mean.device), grad_mean))

                del input_ids
                del attention_mask


for layer in range(32):
    for m in all_modules:
        l_m = "layer_" + str(layer) + "_" + m
        total[l_m] = (total[l_m] / count).cpu()

with open('attpaching/backtracking_entropy_attpatch_AWQ_Llama_8B.pkl', 'wb') as f:
    pickle.dump(total, f)

for h in handles:
    h.remove()

