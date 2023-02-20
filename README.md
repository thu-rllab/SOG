# SOG
Code for NeurIPS2022 accepted paper: Self-Organized Group for Cooperative Multi-agent Reinforcement Learning.

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework and the codebase of [REFIL](https://github.com/shariqiqbal2810/REFIL) algorithm. Thanks for [Shariq Iqbal](https://github.com/shariqiqbal2810) for sharing his code.

## Setup instructions

Please follow the instructions in [REFIL](https://github.com/shariqiqbal2810/REFIL) codebase. Note: If you want to run environment `sc2custom`, an empty map  needs to be copied to the SC2 directory. Note: The particle environment uses gym==0.10.5.

## Run an experiment 

Run an `ALGORITHM` from the folder `src/config/algs`
in an `ENVIRONMENT` from the folder `src/config/envs`

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=<ENVIRONMENT> --config=<ALGORITHM> with <PARAMETERS>
```

Possible ENVIRONMENTs are:
- `particle`: Resource collection and Predator-prey environment from the  paper. The default is Resource collection. Add option `env_args.scenario_id="predator_prey.py"` for using the Predator-prey environment.
- `sc2custom`: StarCraft environment from the paper
  
Possible ALGORITHMs are:
- `rlsocom`: SOG-rl in the paper (recommended);
- `dppsocom`: SOG-dpp in the paper;
- `socom`: SOG in the paper;
- `copa`: COPA in the paper;
- `qmix_atten`: A-QMIX in the paper;
- `qmix_atten_gat`: MAGIC in Appendix I;
- `qmix_atten_silgat`: Gated-ACML in Appendix I.
## Command examples

  Run SOG-rl with environment Resource collection:

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=particle --config=rlsocom with train_map_num=[2,3,4,5] test_map_num=[6,7,8]
```

If you want to use this repository, please consider citing:
```
@inproceedings{shaoself,
  title={Self-Organized Group for Cooperative Multi-agent Reinforcement Learning},
  author={Shao, Jianzhun and Lou, Zhiqiang and Zhang, Hongchang and Jiang, Yuhang and He, Shuncheng and Ji, Xiangyang},
  booktitle={Advances in Neural Information Processing Systems}
}
```