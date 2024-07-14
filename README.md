# RLLR: Enhancing Reinforcement Learning with Label-Sensitive Reward for Natural Language Understanding

This repository contains code for our paper accept at ACL 2024 Main: Enhancing Reinforcement Learning with Label-Sensitive Reward for Natural Language Understanding

Arxiv: https://arxiv.org/abs/2405.19763

## Data

The dataset we used in our experiments can be obtained from the following link:

https://pan.baidu.com/s/13FUopxgGjoisahIVK4u0uA?pwd=cvmq

After downloading the files, modify the paths in `data/dataset_info.json` to your local paths before you run the training.

## Training

The training scripts we used in the experiment are stored in `scripts/`. Naming convention:

- `sft_${model_name}.sh` for SFT training without rationales.

- `sft_wrat_${model_name}.sh` for SFT training with rationales.

- `rlhf_rm_${model_name}.sh` for RLHF reward model training.

- `rlhf_ppo_${model_name}.sh` for RLHF PPO training.

- `rllr_rm_${model_name}.sh` for RLLR reward model training.

- `rllr_ppo_${model_name}.sh` for RLLR PPO training.

- `rllrmix_ppo_${model_name}.sh` for RLLR-mixed PPO training.

The pretrained models can be obtained from huggingface hub:

| Model | Link |
|:-|:-|
| llama2 | https://huggingface.co/meta-llama/Llama-2-7b-chat-hf |
| llama2-13b | https://huggingface.co/meta-llama/Llama-2-13b-chat-hf |
| baichuan2 | https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat |
| baichuan2-13b | https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat |
| chatglm3 | https://huggingface.co/THUDM/chatglm3-6b |
| mistral | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 |
| bloom3b | https://huggingface.co/bigscience/bloom-3b |
| bloom7b | https://huggingface.co/bigscience/bloom-7b1 |

NOTE: after SFT with rationale training, you should merge the LoRA weights back to the base model, and set `PRETRAINED_MODEL_NAME_OR_PATH` to the new model in reward model and PPO training.

## Evaluation

Please refer to `scripts/predict.sh`. You need to modify the `--template` parameters according to different models:

| Model | Template |
|:-|:-|
| llama2 | `llama2` |
| llama2-13b | `llama2` |
| baichuan2 | `baichuan2` |
| baichuan2-13b | `baichuan2` |
| chatglm3 | `chatglm3` |
| mistral | `mistarl` |
| bloom3b | `default` |
| bloom7b | `default` |

## Citation
```
@article{liao2024enhancing,
  title={Enhancing Reinforcement Learning with Label-Sensitive Reward for Natural Language Understanding},
  author={Liao, Kuo and Li, Shuang and Zhao, Meng and Liu, Liqun and Xue, Mengge and Hu, Zhenyu and Han, Honglin and Yin, Chengguo},
  journal={arXiv preprint arXiv:2405.19763},
  year={2024}
}
```
