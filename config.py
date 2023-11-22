from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from RCDRoneInterface import *


def get_config():

    my_config = DEFAULT_CONFIG_DICT.copy()
    my_config["interface"] = DummyRCDroneInterface
    my_config["time_step_duration"] = 0.05
    my_config["start_obs_capture"] = 0.05
    my_config["time_step_timeout_factor"] = 1.0
    my_config["ep_max_length"] = 100
    my_config["act_buf_len"] = 4
    my_config["reset_act_buf"] = False
    my_config["benchmark"] = True
    my_config["benchmark_polyak"] = 0.2

    return my_config
