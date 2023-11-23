
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.envs import GenericGymEnv
from tmrl.training_offline import TorchTrainingOffline

from ActorCritic import *
from Memory import *
from TrainingAgent import *
from utils import *
from RCDRoneInterface import *
from config import get_config
from threading import Thread
from tmrl.util import partial

import sys

import numpy as np

my_config = get_config()

server_ip = "127.0.0.1"
server_port = 6666
security = None
password = "A Secure Password"
LOG_STD_MAX = 2
LOG_STD_MIN = -20
max_samples_per_episode = 1000
model_history = 10

weights_folder = cfg.WEIGHTS_FOLDER
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
my_run_name = "tutorial"

model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))
model_path_history = str(weights_folder / (my_run_name + "_"))

# my_worker.run(test_episode_interval=10)

epochs = np.inf
rounds = 10
steps = 1000
update_buffer_interval = 100
update_model_interval = 1000
max_training_steps_per_env_step = 2.0
start_training = 500
device = None


def run_worker(worker):
    worker.run(test_episode_interval=10)


def run_trainer(trainer):
    trainer.run()


def initialize_vars(server=False):

    my_server = 0
    if server:
        my_server = Server(security=security,
                           password=password,
                           port=server_port)

    env_cls = partial(GenericGymEnv, id="real-time-gym-v1",
                      gym_kwargs={"config": my_config})

    memory_cls = partial(MyMemory,
                         act_buf_len=my_config["act_buf_len"])

    actor_module_cls = partial(MyActorModule)

    training_agent_cls = partial(MyTrainingAgent,
                                 model_cls=MyActorCriticModule,
                                 gamma=0.99,
                                 polyak=0.995,
                                 alpha=0.2,
                                 lr_actor=1e-3,
                                 lr_critic=1e-3,
                                 lr_entropy=1e-3,
                                 learn_entropy_coef=False,
                                 target_entropy=None)

    training_cls = partial(
        TorchTrainingOffline,
        env_cls=env_cls,
        memory_cls=memory_cls,
        training_agent_cls=training_agent_cls,
        epochs=epochs,
        rounds=rounds,
        steps=steps,
        update_buffer_interval=update_buffer_interval,
        update_model_interval=update_model_interval,
        max_training_steps_per_env_step=max_training_steps_per_env_step,
        start_training=start_training,
        device=device)

    my_trainer = Trainer(
        training_cls=training_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints\

    sample_compressor = my_sample_compressor

    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=sample_compressor,
        device=device,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        max_samples_per_episode=max_samples_per_episode,
        model_path=model_path,
        model_path_history=model_path_history,
        model_history=model_history)

    return my_worker, my_trainer, my_server


def start_server():
    _, _, my_server = initialize_vars(server=True)
    print("Server started")


def start_worker():
    my_worker, _, _ = initialize_vars()
    print("Worker started")
    run_worker(my_worker)


def start_trainer():
    _, my_trainer, _ = initialize_vars()
    print("Trainer started")
    run_trainer(my_trainer)


if __name__ == "__main__":

    # RUN IN THREE CMDS WITH DIFFERENT ARGUMENTS

    match int(sys.argv[1]):
        case 0:
            start_server()
        case 1:
            start_trainer()
        case 2:
            start_worker()
