ModuleNotFoundError: No module named 'aiohttp.frozenlist' #pip install aiohttp==3.7.4
AttributeError: module 'aioredis' has no attribute 'create_redis_pool' #pip install aioredis==1.3.1
dqn_trainer.py # add import os
dqn_config.yaml blueprint: "vehicle.lincoln.mkz_2017" -> blueprint: "vehicle.lincoln.mkz2017"
numpy randint problem #pip install gym==0.21.0


pip install -r requirements.txt

python3 dqn_train.py dqn_example/dqn_config.yaml --overwrite