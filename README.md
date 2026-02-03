# Compression-Effect-Analysis
Interpreting the effects of LLMs compression in ICLR 2026 paper ["When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models"](https://arxiv.org/abs/2504.02010).

Navigation:
[Overview](#overview), 
[Datasets](#datasets),
[Mechanistic Interpretation](#mechanistic-interpretation),
[Citation](#citation)


## Overview
Compression methods, including quantization, distillation, and pruning, improve the computational efficiency of large reasoning models (LRMs). However, existing studies either fail to sufficiently compare all three compression methods on LRMs or lack in-depth interpretation analysis. To precisely locate compression effects on model weights, we adapt difference of means and attribution patching techniques, focusing on the activation of every linear component in compressed LRMs, to interpret fine-grained causal relationships between weights and various reasoning capabilities. This fine-grained interpretation addresses a fundamental question of compression: **which weights are the most important for reasoning?** We provide detailed benchmarking results and analysis in our paper. We also share our example code for interpreting the AWQ quantized R1-Distill-Llama-8B. This shared code can be seamlessly applied to other Llama and Qwen models (e.g., GPTQ quantized Llama) after you save appropriate LLMs with their outputs and update model architecture in code accordingly (e.g., update the number of layers).

## Datasets
The details of our selected datasets are specified in our paper. For interpretation, we choose 30 questions from each of our selected datasets and gather model outputs. In `output`, we release the outputs of 4-bit AWQ quantized R1 distilled Llama-8B in `AWQ_R1_Distill_Llama_8B`. Their corresponding annotattion done by GPT-4o is in `AWQ_R1_Distill_Llama_8B_output_annotated`. The released outputs and annotation serve as the example data that our code needs to use.

## Mechanistic Interpretation
To compute the fine-grained steering vectors for each linear module at every layer, run:

    python compute_u.py

You will obtain the steering vectors with respect to all four of our target reasoning capabilities (backtracking, uncertainty estimation, example testing, and adding knowledge).

Then, to compute the final importance score of each weight matrix, we run attribution patching:

    python compute_attpatching.py

The current `compute_attpatching.py` points to the backtracking behavior, so you will need to modify the code slightly to compute the importance scores of other reasoning behaviors. Note that the attribution patching formula can be GPU memory-intensive, since it involves gradient computation! In practice, an 8B-LLM may require 7–8 A100 GPUs (80 GB each).


## Citation
```bibtex
@misc{zhang2025reasoningmeetscompressionunderstanding,
      title={When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models}, 
      author={Nan Zhang and Eugene Kwek and Yusen Zhang and Ngoc-Hieu Nguyen and Prasenjit Mitra and Rui Zhang},
      year={2025},
      eprint={2504.02010},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.02010}, 
}
```
