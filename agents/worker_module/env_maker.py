import gymnasium as gym

from utils.utils import obs_preprocess


class EnvBase:
    def __init__(self, args):
        self.args = args
        self._env = gym.make(args.env)

    def reset(self):
        obs, _ = self._env.reset()
        return obs_preprocess(obs, self.args.need_conv)

    def action_preprocess(func):
        def _wrapper(self, act):
            action_space = self._env.action_space
            if isinstance(action_space, gym.spaces.Discrete): # 이산 행동 공간
                action = act
            else:
                assert isinstance(action_space, gym.spaces.Box) # 연속 행동 공간
                action = [act]
                
            return func(self, action)

        return _wrapper

    @action_preprocess
    def step(self, act):
        obs, rew, terminated, truncated, _ = self._env.step(act)
        return obs_preprocess(obs, self.args.need_conv), rew, terminated or truncated
