# [ACL 2024] ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

🥳 **Welcome!** This codebase accompanies the paper [*ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models*](https://arxiv.org/abs/2406.04214). 

## 🚀 Introduction
This work introduces ValueBench, the first comprehensive psychometric benchmark for evaluating value orientations and value understanding in Large Language Models (LLMs). ValueBench collects data from 44 established psychometric inventories, encompassing 453 multifaceted value dimensions. We propose an evaluation pipeline grounded in realistic human-AI interactions to probe value orientations, along with novel tasks for evaluating value understanding in an open-ended value space.

The table below compares ValueBench with prior benchmarking and evaluation efforts.

<img src="assets/related_work.png" alt="related work" width="400"/>


### Value Orientations
<img src="assets/value_orientation_pipeline.png" alt="value orientation pipeline" width="400"/>

The evaluation pipeline is exemplified in the figure above. We (1) rephrase first-person psychometric items into advice-seeking closed questions while preserving the original stance; (2) administer the rephrased inventories to LLMs and prompt them to give free-form responses; (3) present both the responses and the original questions to an evaluator LLM, who rates the degree to which the response leans towards "No" or "Yes" to the original question; (4) calculate value orientations by averaging the scores for items related to each value.



### Value Understanding
Coming soon!


## 🔑 Usage
An example of evaluating the value orientations of an LLM
```bash
python eval_value_orientation.py --test_model gpt-3.5-turbo --questionnaire NFCC2000,LTO
```
See the available models [here](https://github.com/Value4AI/ValueBench/blob/main/models/__init__.py) and the available questionnaires [here](https://github.com/Value4AI/ValueBench/blob/main/data/value_orientation.csv).

## Citation
If you find ValueBench useful:

```bibtex
@misc{ren2024valuebench,
      title={ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models}, 
      author={Yuanyi Ren and Haoran Ye and Hanjun Fang and Xin Zhang and Guojie Song},
      year={2024},
      eprint={2406.04214},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```