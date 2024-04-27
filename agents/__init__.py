from .learner_module.ppo.learning import learning as learning_ppo
from .learner_module.impala.learning import learning as learning_impala
from .learner_module.sac.learning import learning as learning_sac  # TODO


def ppo_wrapper(timer):
    def _outer(func):
        def _inner(self, *args, **kwargs):
            return learning_ppo(self, timer, *args, **kwargs)

        return _inner

    return _outer


def impala_wrapper(timer):
    def _outer(func):
        def _inner(self, *args, **kwargs):
            return learning_impala(self, timer, *args, **kwargs)

        return _inner

    return _outer


# TODO
def sac_wrapper(timer):
    def _outer(func):
        def _inner(self, *args, **kwargs):
            return learning_sac(self, timer, *args, **kwargs)

        return _inner

    return _outer
