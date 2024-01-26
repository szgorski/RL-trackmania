from rtgym import DEFAULT_CONFIG_DICT
from LidarInterface import *


def get_config():

    my_config = DEFAULT_CONFIG_DICT.copy()
    my_config["interface"] = TM2020InterfaceLidar
    my_config["time_step_duration"] = 0.05
    my_config["start_obs_capture"] = 0.04
    my_config["time_step_timeout_factor"] = 1.0
    my_config["act_buf_len"] = 2
    my_config["benchmark"] = False
    my_config["wait_on_done"] = True

    return my_config
