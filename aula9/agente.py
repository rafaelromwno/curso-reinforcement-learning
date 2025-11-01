from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import os

log_dir = "./logs"

#Cria o ambiente de treinamento
env = make_vec_env("CarRacing-v3",
                   n_envs=1,
                   wrapper_class=WarpFrame,
                   monitor_dir=os.path.join(log_dir, "monitor"))
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

#Cria o ambiente de avaliacao
env_val = make_vec_env("CarRacing-v3", n_envs=1, wrapper_class=WarpFrame)
env_val = VecFrameStack(env_val, n_stack=4)
env_val = VecTransposeImage(env_val)



#Callback de avaliacao
#eval_freq - cuidado para nao deixar esse valor muito baixo!
eval_freq = 5_000

eval_callback = EvalCallback(
    env_val,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=eval_freq,
    render=False,
    deterministic=True,
    n_eval_episodes=20)

checkpoint_callback = CheckpointCallback(
    save_freq=eval_freq,
    save_path=os.path.join(log_dir, "checkpoint")
)

#Lista de Callbacks
callbackList = CallbackList([checkpoint_callback,
                             eval_callback])



#Inicializando o PPO
#ent_coef - valoriza a exploracao
model = PPO('CnnPolicy',
            env,
            verbose=0,
            ent_coef=0.0075,
            tensorboard_log=os.path.join(log_dir, "tensorboard"))

#Treinando o agente
model.learn(total_timesteps=100_000,
            progress_bar=True,
            callback=callbackList)

#Salvando o modelo
model.save(os.path.join(log_dir, "final_model"))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Final Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()






#Cria o ambiente de avaliacao
env = make_vec_env("CarRacing-v3", n_envs=1, seed=0, wrapper_class=WarpFrame)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

#Carrega o mehlor agente
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = PPO.load(best_model_path, env=env)

mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

#Grava um video
best_model_file_name = "best_model_video"
env = VecVideoRecorder(env,
                       log_dir,
                       video_length=5_000,
                       record_video_trigger=lambda x: x == 0,
                       name_prefix=best_model_file_name)

obs = env.reset()
for _ in range(5_000):
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()