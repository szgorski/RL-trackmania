
from tmrl.memory import TorchMemory
from config import get_config
my_config = get_config()


class MyMemory(TorchMemory):
    def __init__(self,
                 device=None,
                 nb_steps=None,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=32,
                 dataset_path="",
                 act_buf_len=my_config["act_buf_len"]):

        self.act_buf_len = act_buf_len  # length of the action buffer

        super().__init__(device=device,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path)

    def append_buffer(self, buffer):
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod) samples
        """

        # decompose compressed samples into their relevant components:

        list_action = [b[0] for b in buffer.memory]
        list_x_position = [b[1][0] for b in buffer.memory]
        list_y_position = [b[1][1] for b in buffer.memory]
        list_x_target = [b[1][2] for b in buffer.memory]
        list_y_target = [b[1][3] for b in buffer.memory]
        list_reward = [b[2] for b in buffer.memory]
        list_terminated = [b[3] for b in buffer.memory]
        list_truncated = [b[4] for b in buffer.memory]
        list_info = [b[5] for b in buffer.memory]

        # append to self.data in some arbitrary way:

        if self.__len__() > 0:
            self.data[0] += list_action
            self.data[1] += list_x_position
            self.data[2] += list_y_position
            self.data[3] += list_x_target
            self.data[4] += list_y_target
            self.data[5] += list_reward
            self.data[6] += list_terminated
            self.data[7] += list_info
            self.data[8] += list_truncated
        else:
            self.data.append(list_action)
            self.data.append(list_x_position)
            self.data.append(list_y_position)
            self.data.append(list_x_target)
            self.data.append(list_y_target)
            self.data.append(list_reward)
            self.data.append(list_terminated)
            self.data.append(list_info)
            self.data.append(list_truncated)

        # trim self.data in some arbitrary way when self.__len__() > self.memory_size:

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0  # self.data is empty
        result = len(self.data[0]) - self.act_buf_len - 1
        if result < 0:
            return 0  # not enough samples to reconstruct the action buffer
        else:
            return result  # we can reconstruct that many samples

    def get_transition(self, item):
        """
        Args:
            item: int: indices of the transition that the Trainer wants to sample
        Returns:
            full transition: (last_obs, new_act, rew, new_obs, terminated, truncated, info)
        """
        idx_last = item + self.act_buf_len - 1  # index of previous observation
        idx_now = item + self.act_buf_len  # index of new observation

        # rebuild the action buffer of both observations:
        actions = self.data[0][item:(item + self.act_buf_len + 1)]
        last_act_buf = actions[:-1]  # action buffer of previous observation
        new_act_buf = actions[1:]  # action buffer of new observation

        # rebuild the previous observation:
        last_obs = (self.data[1][idx_last],  # x position
                    self.data[2][idx_last],  # y position
                    self.data[3][idx_last],  # x target
                    self.data[4][idx_last],  # y target
                    *last_act_buf)  # action buffer

        # rebuild the new observation:
        new_obs = (self.data[1][idx_now],  # x position
                   self.data[2][idx_now],  # y position
                   self.data[3][idx_now],  # x target
                   self.data[4][idx_now],  # y target
                   *new_act_buf)  # action buffer

        # other components of the transition:
        new_act = self.data[0][idx_now]  # action
        rew = np.float32(self.data[5][idx_now])  # reward
        terminated = self.data[6][idx_now]  # terminated signal
        truncated = self.data[8][idx_now]  # truncated signal
        info = self.data[7][idx_now]  # info dictionary

        return last_obs, new_act, rew, new_obs, terminated, truncated, info
