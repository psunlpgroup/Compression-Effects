import pickle
import re
import unicodedata
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import torch


model_name = "compressed_R1_Llama_8B"
tokenizer = AutoTokenizer.from_pretrained('HF_models/DeepSeek-R1-Distill-Llama-8B')
model = AutoModelForCausalLM.from_pretrained(model_name)


def make_hook(layer_idx: int, proj_name: str):
    def hook(module, inputs, output):
        activations[f"layer_{layer_idx}_{proj_name}"] = torch.squeeze(output.detach())
    return hook

for layer_idx, layer in enumerate(model.model.layers):
    # self_attn projections
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin_mod = getattr(layer.self_attn, proj)
        lin_mod.register_forward_hook(make_hook(layer_idx, proj))
    # mlp projections
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin_mod = getattr(layer.mlp, proj)
        lin_mod.register_forward_hook(make_hook(layer_idx, proj))


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


capabilities = ["backtracking", "uncertainty-estimation", "example-testing", "adding-knowledge"]
all_modules, D_plus_act, D_minus_act, D_plus_count = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], {}, {}, [0] * 4
total_sum = {}
for layer in range(32):
    for m in all_modules:
        l_m = "layer_" + str(layer) + "_" + m
        if m in ["q_proj", "o_proj", "down_proj"]:
            D_plus_act[l_m] = [torch.zeros(4096, device=torch.device('cuda'))] * 4
            D_minus_act[l_m] = torch.zeros(4096, device=torch.device('cuda'))
            total_sum[l_m] = torch.zeros(4096, device=torch.device('cuda'))
        elif m in ["k_proj", "v_proj"]:
            D_plus_act[l_m] = [torch.zeros(1024, device=torch.device('cuda'))] * 4
            D_minus_act[l_m] = torch.zeros(1024, device=torch.device('cuda'))
            total_sum[l_m] = torch.zeros(1024, device=torch.device('cuda'))
        elif m in ["gate_proj", "up_proj"]:
            D_plus_act[l_m] = [torch.zeros(14336, device=torch.device('cuda'))] * 4
            D_minus_act[l_m] = torch.zeros(14336, device=torch.device('cuda'))
            total_sum[l_m] = torch.zeros(14336, device=torch.device('cuda'))
total_len = 0



##### main loop #######
for z in range(len(all_d)):
    for j in range(30):
        print(j)
        with open("output/AWQ_R1_Distill_Llama_8B/" + all_d[z] + "/" + str(j) + ".txt", "r", encoding="utf-8") as f:
            original = total_input_all[z][j] + f.read()

        with open("output/AWQ_R1_Distill_Llama_8B_output_annotated/" + all_d[z] + "/" + str(j) + ".txt", "r", encoding="utf-8") as f:
            annotated = f.read()

        activations = {}
        inputs = tokenizer(original, return_tensors="pt")
        input_ids = inputs["input_ids"]
        _ = model(input_ids, use_cache=False)

        total_len += activations['layer_0_q_proj'].size(0)
        for layer in range(32):
            for m in all_modules:
                l_m = "layer_" + str(layer) + "_" + m
                D_minus_act[l_m] += torch.mean(activations[l_m], 0).cuda()

                total_sum[l_m] += activations[l_m].sum(dim=0).cuda()



        labels = [
            "initializing", "deduction", "adding-knowledge",
            "example-testing", "uncertainty-estimation", "backtracking",
        ]
        seg_re = re.compile(r'\["(' + "|".join(labels) + r')"\]\s*(.*?)\s*\["end-section"\]', re.DOTALL)
        segments = [{"label": m.group(1), "text": m.group(2).strip()} for m in seg_re.finditer(annotated)]

        tokenizer2 = AutoTokenizer.from_pretrained('HF_models/DeepSeek-R1-Distill-Llama-8B', use_fast=True)
        enc = tokenizer2(
            original,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = enc.offset_mapping
        if len(enc['input_ids']) != activations['layer_0_q_proj'].size()[0]:
            print("sequence dimension mismatch!!!")
            break

        threshold = 0.90

        # -----------------------------------------------------------------------------
        # Reasoning capabilities loop
        # -----------------------------------------------------------------------------
        for c in range(4):
            print(capabilities[c])
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

                if seg['label'] == capabilities[c]:
                    flag = True
                    target_indices.append([start_tok, end_tok])
            if flag:
                D_plus_count[c] += 1
                rows = []
                for indices in target_indices:
                    rows += list(range(indices[0]-5, indices[1]))
                for layer in range(32):
                    for m in all_modules:
                        l_m = "layer_" + str(layer) + "_" + m
                        D_plus_act[l_m][c] += torch.mean(activations[l_m][rows, :], 0).cuda()


        del activations

names = ["backtracking", "uncertainty", "example", "knowledge"]
for c in range(4):
    u = {}
    for layer in range(32):
        for m in all_modules:
            l_m = "layer_" + str(layer) + "_" + m
            temp_u = D_plus_act[l_m][c] / D_plus_count[c] - D_minus_act[l_m] / 120
            a_overall_mean = total_sum[l_m] / total_len
            u[l_m] = temp_u * (torch.norm(a_overall_mean, p=2) / torch.norm(temp_u, p=2))
            u[l_m] = u[l_m].cpu()


    with open('all_u/' + names[c] + '_u_AWQ_Llama_8B_new.pkl', 'wb') as f:
        pickle.dump(u, f)


