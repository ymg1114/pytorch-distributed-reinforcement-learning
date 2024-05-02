import gym

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
            action = [act] if "Pendulum" in self._env.spec.name else act
            return func(self, action)

        return _wrapper

    @action_preprocess
    def step(self, act):
        obs, rew, done, _, _ = self._env.step(act)
        return obs_preprocess(obs, self.args.need_conv), rew, done
