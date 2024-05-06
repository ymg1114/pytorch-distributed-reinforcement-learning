from .learner_module.ppo.learning import learning as alearning_ppo
from .learner_module.v_mpo.learning import learning as alearning_v_mpo
from .learner_module.impala.learning import learning as alearning_impala
from .learner_module.sac.learning import learning as alearning_sac
from .learner_module.sac_continuous.learning import learning as alearning_sac_continuous


def ppo_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_ppo(self, timer, *args, **kwargs)

        return _inner

    return _outer


def v_mpo_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_v_mpo(self, timer, *args, **kwargs)

        return _inner

    return _outer


def impala_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_impala(self, timer, *args, **kwargs)

        return _inner

    return _outer


def sac_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_sac(self, timer, *args, **kwargs)

        return _inner

    return _outer


def sac_continuous_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_sac_continuous(self, timer, *args, **kwargs)

        return _inner

    return _outer
