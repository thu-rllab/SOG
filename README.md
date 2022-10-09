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

Possible environments are:
- `particle`: Resource collection and Predator-prey environment from the  paper. The default is Resource collection. Add option `env_args.scenario_id="predator_prey.py"` for using the Predator-prey environment.
- `sc2custom`: StarCraft environment from the paper

## Command examples

  Run SOG with environment Resource collection:

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=particle --config=socom with train_map_num=[2,3,4,5] test_map_num=[6,7,8]
```

Run SOG with Predator-prey:

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=particle --config=socom with env_args.scenario_id="predator_prey.py" train_map_num=[[3,4],[1]] test_map_num=[[5,6],[1,2]]
```

Run SOG on StarCraft map 3-8sz_symmetric_G(ather):

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=sc2custom --config=config with scenario=3-8sz_symmetric train_map_num=[3,4,5] test_map_num=[6,7,8]
```

Run SOG on StarCraft map 3-8MMM_symmetric_D(isperse):

```shell
export CUDA_VISIBLE_DEVICES="0" && python src/main.py --env-config=sc2custom --config=config with scenario=3-8MMM_symmetric train_map_num=[3,4,5] test_map_num=[6,7,8] env_args.divide_group=True env_args.sight_range=3
```

If you want to use this repository, please consider cite:
```
@inproceedings{
anonymous2022selforganized,
title={Self-Organized Group for Cooperative Multi-agent Reinforcement Learning},
author={Shao, Jianzhun and Lou, Zhiqiang and Zhang, Hongchang and Jiang, Yuhang and He, Shuncheng and Ji, Xiangyang},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=hd5KRowT3oB}
}
```