import os
import asyncio
import time
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from utils.utils import ExecutionTimer

from ..compute_loss import compute_v_trace


async def learning(parent, timer: ExecutionTimer):
    assert hasattr(parent, "batch_queue")  # asyncio.Queue(buffer_size)
    parent.idx = 0

    while True:
        batch_args = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_args = await parent.batch_queue.get()

            if batch_args is not None:
                with timer.timer("learner-forward-time"):
                    # Basically, mini-batch-learning (batch, seq, feat)
                    obs, act, rew, logits, is_fir, hx, cx = parent.to_gpu(*batch_args)
                    behav_log_probs = (
                        parent.CT(F.softmax(logits, dim=-1))
                        .log_prob(act.squeeze(-1))
                        .unsqueeze(-1)
                    )

                    # epoch-learning
                    for _ in range(parent.args.K_epoch):
                        lstm_states = (
                            hx[:, 0],
                            cx[:, 0],
                        )  # (batch, seq, hidden) -> (batch, hidden)

                        # on-line model forwarding
                        log_probs, entropy, value = parent.model.actor(
                            obs,  # (batch, seq, *sha)
                            lstm_states,  # ((batch, hidden), (batch, hidden))
                            act,  # (batch, seq, 1)
                        )

                        # V-trace를 사용하여 off-policy corrections 연산
                        ratio, advantages, values_target = compute_v_trace(
                            behav_log_probs=behav_log_probs,
                            target_log_probs=log_probs,
                            is_fir=is_fir,
                            rewards=rew,
                            values=value,
                            gamma=parent.args.gamma,
                        )

                        loss_policy = -(log_probs[:, :-1] * advantages).mean()
                        loss_value = F.smooth_l1_loss(
                            value[:, :-1], values_target[:, :-1]
                        ).mean()
                        policy_entropy = entropy[:, :-1].mean()

                        loss = (
                            parent.args.policy_loss_coef * loss_policy
                            + parent.args.value_loss_coef * loss_value
                            - parent.args.entropy_coef * policy_entropy
                        )

                        detached_losses = {
                            "policy-loss": loss_policy.detach().cpu(),
                            "value-loss": loss_value.detach().cpu(),
                            "policy-entropy": policy_entropy.detach().cpu(),
                            "ratio": ratio.detach().cpu(),
                        }

                with timer.timer("learner-backward-time"):
                    parent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parent.model.actor.parameters(), parent.args.max_grad_norm
                    )
                    print(
                        "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                            loss.item(),
                            detached_losses["value-loss"],
                            detached_losses["policy-loss"],
                            detached_losses["policy-entropy"],
                            detached_losses["ratio"].mean(),
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
