
def my_sample_compressor(act, obs, rew, terminated, truncated, info):
    """
    Compresses samples before sending them over the network.

    This function creates the sample that will actually be stored in local buffers for networking.
    This is to compress the sample before sending it over the Internet/local network.
    Buffers of compressed samples will be given as input to the append() method of the memory.
    When you implement a compressor, you also need to implement a decompressor in the memory.

    Args:
        act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    Returns:
        act_mod: compressed act
        obs_mod: compressed obs
        rew_mod: compressed rew
        terminated_mod: compressed terminated
        truncated_mod: compressed truncated
        info_mod: compressed info
    """
    act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod = act, obs, rew, terminated, truncated, info
    obs_mod = obs_mod[:4]  # here we remove the action buffer from observations
    return act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod
