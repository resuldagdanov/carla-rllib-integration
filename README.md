```sh
conda create -n rllib python=3.7

conda activate rllib

pip install -r requirements.txt

Download Carla 0.9.11 https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
```

```sh
gedit ~/.bashrc

export CARLA_ROOT=PATH_TO_CARLA_ROOT
export SCENARIO_RUNNER_ROOT=PATH_TO_SCENARIO_RUNNER
export LEADERBOARD_ROOT=PATH_TO_LEADERBOARD
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg":${PYTHONPATH}

source ~/.bashrc
```

```sh
python3 dqn_train.py dqn_example/dqn_config.yaml --overwrite
```

## SOLVED PROBLEMS

ModuleNotFoundError: No module named 'aiohttp.frozenlist' #pip install aiohttp==3.7.4

AttributeError: module 'aioredis' has no attribute 'create_redis_pool' #pip install aioredis==1.3.1

dqn_trainer.py # add import os

dqn_config.yaml blueprint: "vehicle.lincoln.mkz_2017" -> blueprint: "vehicle.lincoln.mkz2017"

numpy randint problem #pip install gym==0.21.0