# Llama_Slayer
Official code for our paper "Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection", accepted by EMNLP Findings 2024.
[https://aclanthology.org/2024.findings-emnlp.347.pdf]

## Highlight

* A novel knowledge injection strategy for Large Language Models (LLMs) during the continual pretraining phase. This approach delivers substantial performance improvements in domain-specific knowledge but with significantly fewer trainable parameters—only 25% of those required for full fine-tuning. Additionally, it effectively mitigates Catastrophic Forgetting better than full fine-tuning.

## News

* 24-11-13. Our code for layer expansion has been uploaded. You may choose to remove the final x layers or not.

## Overview of our layer expansion strategy

![image](https://github.com/txchen-USTC/Llama-Slayer/blob/main/asset/strategy.jpg)

* Expand within the first half layers, since injecting domain-specific knowledge to the shallow layers are more effective than even injection according to our finding.
* Weight averaging as the initialization of the expanded layers to better curtail catastrophic forgetting, instead of direct copying the weights of the former layer.
* Remove the final few layers (~2 for 32-layer 7B models) and the impact is trivial.


## Citation

Please cite our paper if you find the repository helpful.
```
@inproceedings{chen2024llama,
  title={Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection},
  author={Chen, Tianxiang and Tan, Zhentao and Gong, Tao and Wu, Yue and Chu, Qi and Liu, Bin and Ye, Jieping and Yu, Nenghai},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={5991--6002},
  year={2024}
}
```
