#!/usr/bin/env python3
"""1次元ガウス分布での Noise-Contrastive Estimation (NCE) の Toy Example."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 7
    n_data: int = 2_000
    noise_ratio: int = 5  # k: noise/data
    n_steps: int = 2_000
    batch_size: int = 128
    lr: float = 0.03

    mu_true: float = 2.0
    sigma_model: float = 1.0

    mu_noise: float = 0.0
    sigma_noise: float = 3.0


def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * math.log(2.0 * math.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def sample_normal(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)


def nce_train(data: list[float], cfg: Config) -> tuple[float, float]:
    """NCE で (mu, c) を推定する。"""

    mu = 0.0
    c = 0.0
    sigma2 = cfg.sigma_model**2

    for step in range(cfg.n_steps):
        # データバッチ
        x_d = [data[random.randrange(len(data))] for _ in range(cfg.batch_size)]

        # ノイズバッチ（k倍）
        x_n = [
            sample_normal(cfg.mu_noise, cfg.sigma_noise)
            for _ in range(cfg.batch_size * cfg.noise_ratio)
        ]

        sum_grad_mu_d = 0.0
        sum_grad_c_d = 0.0
        sum_log_pd = 0.0

        for x in x_d:
            log_p_tilde = -0.5 * ((x - mu) ** 2) / sigma2
            log_kq = math.log(cfg.noise_ratio) + log_normal_pdf(x, cfg.mu_noise, cfg.sigma_noise)
            g = log_p_tilde - c - log_kq
            p = sigmoid(g)

            dg = p - 1.0  # y=1
            dmu = (x - mu) / sigma2
            sum_grad_mu_d += dg * dmu
            sum_grad_c_d += dg * (-1.0)
            sum_log_pd += math.log(p + 1e-12)

        sum_grad_mu_n = 0.0
        sum_grad_c_n = 0.0
        sum_log_1mn = 0.0

        for x in x_n:
            log_p_tilde = -0.5 * ((x - mu) ** 2) / sigma2
            log_kq = math.log(cfg.noise_ratio) + log_normal_pdf(x, cfg.mu_noise, cfg.sigma_noise)
            g = log_p_tilde - c - log_kq
            p = sigmoid(g)

            dg = p - 0.0  # y=0
            dmu = (x - mu) / sigma2
            sum_grad_mu_n += dg * dmu
            sum_grad_c_n += dg * (-1.0)
            sum_log_1mn += math.log(1.0 - p + 1e-12)

        # NCE目的: data 1件に対して noise k件をそのまま足し込む
        grad_mu = (sum_grad_mu_d + sum_grad_mu_n) / len(x_d)
        grad_c = (sum_grad_c_d + sum_grad_c_n) / len(x_d)

        mu -= cfg.lr * grad_mu
        c -= cfg.lr * grad_c

        if step % 400 == 0 or step == cfg.n_steps - 1:
            loss = -(sum_log_pd + sum_log_1mn) / len(x_d)
            print(f"step={step:4d} loss={loss:.4f} mu={mu:.4f} c={c:.4f}")

    return mu, c


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)

    data = [sample_normal(cfg.mu_true, cfg.sigma_model) for _ in range(cfg.n_data)]

    mu_mle = sum(data) / len(data)  # 正規分布(分散既知)なら MLE は標本平均
    mu_nce, c_nce = nce_train(data, cfg)

    # 真の logZ（この設定なら解析的に求まる）
    # log p_tilde = - (x-mu)^2/(2sigma^2) なので
    # Z = ∫ exp(log p_tilde) dx = sqrt(2*pi)*sigma
    logz_true = 0.5 * math.log(2.0 * math.pi * cfg.sigma_model**2)

    print("\n=== Result ===")
    print(f"true mu        : {cfg.mu_true:.4f}")
    print(f"MLE  mu        : {mu_mle:.4f}")
    print(f"NCE  mu        : {mu_nce:.4f}")
    print(f"true logZ      : {logz_true:.4f}")
    print(f"NCE  c (logZ)  : {c_nce:.4f}")


if __name__ == "__main__":
    main()
