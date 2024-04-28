import os
import asyncio
import time
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from utils.utils import ExecutionTimer

from ..compute_loss import soft_update


async def learning(parent, timer: ExecutionTimer):
    assert hasattr(parent, "batch_buffer")
    parent.idx = 0

    while True:
        batch_args = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_args = await parent.batch_buffer.get()

            if batch_args is not None:
                with timer.timer("learner-forward-time"):
                    # Basically, mini-batch-learning (batch, seq, feat)
                    obs, act, rew, logits, is_fir, hx, cx = parent.to_gpu(*batch_args)

                    # epoch-learning
                    for _ in range(parent.args.K_epoch):
                        lstm_states = (
                            hx[:, 0],
                            cx[:, 0],
                        )  # (batch, seq, hidden) -> (batch, hidden)

                        # 현재 policy에서 샘플링, only to calculate "loss_policy"
                        act_probs_pol, log_probs_pol = parent.actor(
                            obs,  # (batch, seq, *sha)
                            lstm_states,  # ((batch, hidden), (batch, hidden))
                        )
                        q1_pol, q2_pol = parent.critic(obs, lstm_states)
                        min_q_pol = torch.min(q1_pol, q2_pol)  # q 값 과대평가 방지

                        # 샘플링 확률 (act_probs_pol)을 곱하여 기대값 반영
                        loss_policy = (
                            (
                                act_probs_pol
                                * (parent.args.alpha * log_probs_pol - min_q_pol)
                            )[:, :-1]
                            .sum(-1)
                            .mean()
                        )

                        parent.actor_optimizer.zero_grad()
                        loss_policy.backward()
                        torch.nn.utils.clip_grad_norm_(
                            parent.actor.parameters(), parent.args.max_grad_norm
                        )
                        parent.actor_optimizer.step()

                        with torch.no_grad():
                            # 현재 policy에서 샘플링, only to calculate "td_target"
                            act_probs_cri, log_probs_cri = parent.actor(
                                obs,  # (batch, seq, *sha)
                                lstm_states,  # ((batch, hidden), (batch, hidden))
                            )
                            tar_q1_cri, tar_q2_cri = parent.target_critic(
                                obs, lstm_states
                            )
                            min_tar_q_cri = torch.min(
                                tar_q1_cri, tar_q2_cri
                            )  # q 값 과대평가 방지

                            # 샘플링 확률 (act_probs_cri)을 곱하여 기대값 반영
                            sampled_soft_q = act_probs_cri * (
                                min_tar_q_cri - parent.args.alpha * log_probs_cri
                            )

                            # Target Soft Q-value
                            td_target = rew[:, :-1] + (
                                1 - is_fir[:, 1:]
                            ) * parent.args.gamma * sampled_soft_q[:, 1:].sum(
                                dim=-1
                            ).unsqueeze(
                                -1
                            )

                        src_q1, src_q2 = parent.critic(obs, lstm_states)  # 현재 critic

                        src_q1 = src_q1.gather(-1, act.long())
                        src_q2 = src_q2.gather(-1, act.long())

                        loss_value = (
                            F.smooth_l1_loss(src_q1[:, :-1], td_target).mean()
                            + F.smooth_l1_loss(src_q2[:, :-1], td_target).mean()
                        )

                        parent.critic_optimizer.zero_grad()
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            parent.critic.parameters(), parent.args.max_grad_norm
                        )
                        parent.critic_optimizer.step()

                        loss = (
                            parent.args.policy_loss_coef * loss_policy
                            + parent.args.value_loss_coef * loss_value
                        )

                        detached_losses = {
                            "policy-loss": loss_policy.detach().cpu(),
                            "value-loss": loss_value.detach().cpu(),
                        }

                        print(
                            "original_value_loss: {:.5f} original_policy_loss: {:.5f}".format(
                                detached_losses["value-loss"],
                                detached_losses["policy-loss"],
                            )
                        )

                        soft_update(parent.critic, parent.target_critic)

                parent.pub_model(parent.actor.state_dict())

                if parent.idx % parent.args.loss_log_interval == 0:
                    parent.log_loss_tensorboard(timer, loss, detached_losses)

                if parent.idx % parent.args.model_save_interval == 0:
                    torch.save(
                        parent.actor,
                        os.path.join(
                            parent.args.model_dir, f"{parent.args.algo}_{parent.idx}.pt"
                        ),
                    )

                parent.idx += 1

            if parent.heartbeat is not None:
                parent.heartbeat.value = time.time()

        await asyncio.sleep(0.001)
