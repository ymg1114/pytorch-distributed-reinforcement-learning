import os
import asyncio
import time
import math
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from utils.utils import ExecutionTimer

from ..compute_loss import compute_gae, kldivergence


async def learning(parent, timer: ExecutionTimer):
    assert hasattr(parent, "batch_queue")
    parent.idx = 0

    while not parent.stop_event.is_set():
        batch_args = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_args = await parent.batch_queue.get()

            if batch_args is not None:
                with timer.timer("learner-forward-time"):
                    # Basically, mini-batch-learning (batch, seq, feat)
                    obs, act, rew, behave_logits, _, is_fir, hx, cx = parent.to_gpu(
                        *batch_args
                    )
                    # behav_log_probs = (
                    #     parent.CT(F.softmax(logits, dim=-1))
                    #     .log_prob(act.squeeze(-1))
                    #     .unsqueeze(-1)
                    # )

                    # epoch-learning
                    for _ in range(parent.args.K_epoch):
                        lstm_states = (
                            hx[:, 0],
                            cx[:, 0],
                        )  # (batch, seq, hidden) -> (batch, hidden)

                        # on-line model forwarding
                        logits, log_probs, _, value = parent.model.actor(
                            obs,  # (batch, seq, *sha)
                            lstm_states,  # ((batch, hidden), (batch, hidden))
                            act,  # (batch, seq, 1)
                        )
                        with torch.no_grad():
                            td_target = (
                                rew[:, :-1]
                                + parent.args.gamma * (1 - is_fir[:, 1:]) * value[:, 1:]
                            )
                            delta = td_target - value[:, :-1]

                            gae = compute_gae(
                                delta, parent.args.gamma, parent.args.lmbda
                            )  # gae (advantage)

                            top_gae, top_idx = (
                                torch.topk(  # top 50% in batch-dim (not, seq-dim)
                                    gae, math.ceil(gae.size(0) / 2), 0
                                )
                            )

                            ratio = top_gae / (parent.log_eta.exp().detach() + 1e-7)

                        top_log_probs = log_probs[:, :-1].gather(0, top_idx)

                        psi = F.softmax(
                            ratio.view(-1), 0
                        )  # 평탄화 하고 -> softmax -> 쉐잎 원복
                        # psi.reshape(ratio.shape) == ratio.exp() / (ratio.exp().sum() + 1e-7)
                        loss_policy = -(psi.reshape(ratio.shape) * top_log_probs).sum()

                        loss_value = (
                            F.smooth_l1_loss(  # TODO: v-mpo는 조금 다른 거 같은데..
                                value[:, :-1], td_target
                            ).mean()
                        )

                        loss_temperature = (
                            parent.log_eta.exp() * parent.args.coef_eta
                            + parent.log_eta.exp() * (ratio.exp().mean().log())
                        )

                        alpha = parent.log_alpha.exp()
                        Kl = kldivergence(behave_logits[:, :-1], logits[:, :-1])
                        loss_alpha = (
                            alpha * (parent.get_coef_alpha() - Kl.detach())
                            + alpha.detach() * Kl
                        ).mean()

                        loss = (
                            parent.args.policy_loss_coef * loss_policy
                            + parent.args.value_loss_coef * loss_value
                            + loss_temperature
                            + loss_alpha
                        )

                        detached_losses = {
                            "policy-loss": loss_policy.detach().cpu(),
                            "value-loss": loss_value.detach().cpu(),
                            "loss-temperature": loss_temperature.detach().cpu(),
                            "loss-alpha": loss_alpha.detach().cpu(),
                        }

                        with timer.timer("learner-backward-time"):
                            parent.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                parent.model.actor.parameters(),
                                parent.args.max_grad_norm,
                            )
                            print(
                                "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} loss-temperature: {:.5f} loss-alpha: {:.5f}".format(
                                    loss.item(),
                                    detached_losses["value-loss"],
                                    detached_losses["policy-loss"],
                                    detached_losses["loss-temperature"],
                                    detached_losses["loss-alpha"],
                                )
                            )
                            parent.optimizer.step()

                parent.pub_model(parent.model.actor.state_dict())

                if parent.idx % parent.args.loss_log_interval == 0:
                    parent.log_loss_tensorboard(timer, loss, detached_losses)

                if parent.idx % parent.args.model_save_interval == 0:
                    torch.save(
                        parent.model,
                        os.path.join(
                            parent.args.model_dir, f"{parent.args.algo}_{parent.idx}.pt"
                        ),
                    )

                parent.idx += 1

            if parent.heartbeat is not None:
                parent.heartbeat.value = time.time()

        await asyncio.sleep(0.001)
