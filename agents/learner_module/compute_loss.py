import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


def compute_gae(
    deltas,
    gamma,
    lambda_,
):
    gae = 0
    returns = []
    for t in reversed(range(deltas.size(1))):
        d = deltas[:, t]
        gae = d + gamma * lambda_ * gae
        returns.insert(0, gae)

    return torch.stack(returns, dim=1)


def compute_v_trace(
    behav_log_probs,
    target_log_probs,
    is_fir,
    rewards,
    values,
    gamma,
    rho_bar=0.8,
    c_bar=1.0,
):
    # Importance sampling weights (rho)
    rho = torch.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    ).detach()  # a/b == exp(log(a)-log(b))
    # rho_clipped = torch.clamp(rho, max=rho_bar)
    rho_clipped = torch.clamp(rho, min=0.1, max=rho_bar)

    # truncated importance weights (c)
    c = torch.exp(
        target_log_probs[:, :-1] - behav_log_probs[:, :-1]
    ).detach()  # a/b == exp(log(a)-log(b))
    c_clipped = torch.clamp(c, max=c_bar)

    td_target = rewards[:, :-1] + gamma * (1 - is_fir[:, 1:]) * values[:, 1:]
    deltas = rho_clipped * (td_target - values[:, :-1])  # TD-Error with 보정

    vs_minus_v_xs = torch.zeros_like(values)
    for t in reversed(range(deltas.size(1))):
        vs_minus_v_xs[:, t] = (
            deltas[:, t]
            + c_clipped[:, t]
            * (gamma * (1 - is_fir))[:, t + 1]
            * vs_minus_v_xs[:, t + 1]
        )

    values_target = (
        values + vs_minus_v_xs
    )  # vs_minus_v_xs는 V-trace를 통해 수정된 가치 추정치
    advantages = rho_clipped * (
        rewards[:, :-1]
        + gamma * (1 - is_fir[:, 1:]) * values_target[:, 1:]
        - values[:, :-1]
    )

    return rho_clipped, advantages.detach(), values_target.detach()


def soft_update(critic, target_critic, tau=0.005):
    for p, target_p in zip(critic.parameters(), target_critic.parameters()):
        target_p.data.copy_((1.0 - tau) * target_p.data + tau * p.data)


def kldivergence(logits_p, logits_q):
    dist_p = Categorical(F.softmax(logits_p, dim=-1))
    dist_q = Categorical(F.softmax(logits_q, dim=-1))
    return kl_divergence(dist_p, dist_q).squeeze()
