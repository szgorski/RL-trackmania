"""
==============================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
==============================================================================

In this tutorial, we will customize the TrackMania training pipeline.

The tutorial works with the TrackMania FULL Gymnasium environment.
Please refer to the README on GitHub to set up this environment in config.json:
https://github.com/trackmania-rl/tmrl#full-environment

Note: This tutorial describes implementing and running a TrainingAgent along with an ActorModule.
It is relevant if you want to implement your own RL approaches in TrackMania.
If you plan to try non-RL approaches instead, this is also accepted:
just use the competition Gymnasium Full environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit your entry :)

Copy and adapt this script to implement your own algorithm/model in TrackMania.
Then, use the script as follows:

To launch the Server, provided the script is named custom_actor_module.py, execute:
python custom_actor_module.py --server

In another terminal, launch the Trainer:
python custom_actor_module.py --trainer

And in yet another terminal, launch a RolloutWorker:
python custom_actor_module.py --worker

You can launch these in any order, but we recommend server, then trainer, then worker.
If you are running everything on the same machine, your trainer may consume all your resource,
resulting in your worker struggling to collect samples in a timely fashion.
If your worker crazily warns you about time-steps timing out, this is probably the issue.
The best way of using TMRL with TrackMania is to have your worker(s) and trainer on separate machines.
The server can run on either of these machines, or yet another machine that both can reach via network.
Achieving this is easy (and is also kind of the whole point of the TMRL framework).
Just adapt config.json (or this script) to your network configuration.
In particular, you will want to set the following in the TMRL config.json file of all your machines:

"LOCALHOST_WORKER": false,
"LOCALHOST_TRAINER": false,
"PUBLIC_IP_SERVER": "<ip.of.the.server>",
"PORT": <port of the server (usually requires port forwarding if accessed via the Internet)>,

If you are training over the Internet, please read the security instructions on the TMRL GitHub page.
"""

# Let us start our tutorial by importing some useful stuff.

# The constants that are defined in config.json:
from torch.optim import Adam
import itertools
from tmrl.util import cached_property
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.training import TrainingAgent
import json
from math import floor
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn
import torch
from tmrl.actor import ActorModule
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
from tmrl.util import prod

# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
from copy import deepcopy
import numpy as np
import os


# Now, let us look into the content of config.json:

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly (not recommended),
# or you can change them in the TMRL config.json file (recommended).

# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch', this is also when training metrics can be logged to wandb)
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
# (Change this with your own if you want to keep your training curves private)
# (Also, please use your own wandb account if you are going to log huge stuff :) )

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
# name of the wandb project in which your run will appear
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

# this line sets your wandb API key as the active key
os.environ['WANDB_API_KEY'] = wandb_key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY

memory_base_cls = cfg_obj.MEM

sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

sample_preprocessor = None

dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

env_cls = cfg_obj.ENV_CLS
device_worker = 'cpu'


window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT
img_grayscale = cfg.GRAYSCALE

imgs_buf_len = cfg.IMG_HIST_LEN
act_buf_len = cfg.ACT_BUF_LEN

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (
        conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (
        conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(ActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape)
                      for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        tensors = [torch.from_numpy(obs[0]), torch.from_numpy(obs[1]), torch.from_numpy(
            obs[2]), torch.from_numpy(obs[3])]
        net_out = self.net(torch.cat(tensors, -1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action -
                        F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(
            observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(
            observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.numpy()


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class SACTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC in this tutorial).

    Custom TrainingAgents implement two methods: train(batch) and get_actor().
    The train method performs a training step.
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.

    Your implementation must also pass three required arguments to the superclass:

    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(
        lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 # Gymnasium observation space (required argument here for your convenience)
                 observation_space=None,
                 # Gymnasium action space (required argument here for your convenience)
                 action_space=None,
                 # Device our TrainingAgent should use for training (required argument)
                 device=None,
                 # An actor-critic module, encapsulating our ActorModule
                 model_cls=MLPActorCritic,
                 gamma=0.99,  # Discount factor
                 polyak=0.995,  # Exponential averaging factor for the target critic
                 alpha=0.2,  # Value of the entropy coefficient
                 lr_actor=1e-3,  # Learning rate for the actor
                 lr_critic=1e-3):  # Learning rate for the critic

        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        # custom stuff:
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.q_params = itertools.chain(
            self.model.q1.parameters(), self.model.q2.parameters())
        # self.pi_optimizer = Adam(
        #   self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit

        Note that we will ignore the t signal for a clever reason.
        When the episode is truncated due to a time limit, we do not want our model to believe that o2 is terminal.
        Instead, we will make our model believe that the episode would normally have continued after this transition.
        Because we use a discount factor, this does not result in an infinite value.
        In our situation, the discount factor is what incentivizes the AI to run as fast as possible!

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        # We sample an action in the current policy and retrieve its corresponding log probability:

        pi, logp_pi = self.model.actor(observation_space=o, action_space=a)

        # We also compute our action-value estimates for the current transition:
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        # Now we compute our value target, for which we need to detach from gradients computation:
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2, a)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict


training_agent_cls = partial(SACTrainingAgent,
                             model_cls=MLPActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=0.000005,
                             lr_critic=0.00003)


training_cls = partial(
    TrainingOffline,
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
    device=device_trainer)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true',
                        help='launches the server')
    parser.add_argument('--trainer', action='store_true',
                        help='launches the trainer')
    parser.add_argument('--worker', action='store_true',
                        help='launches a rollout worker')
    parser.add_argument('--test', action='store_true',
                        help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run_with_wandb(entity=wandb_entity,
                                  project=wandb_project,
                                  run_id=wandb_run_id)
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=SquashedGaussianMLPActor,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run()
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)
