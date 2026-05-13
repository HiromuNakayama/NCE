set.seed(123)

############################################################
# 0. Basic settings
############################################################

K <- 2
T_len <- 3000

A_true <- matrix(
  c(0.95, 0.05,
    0.10, 0.90),
  nrow = K,
  byrow = TRUE
)

# log \tilde p_k(x)
# = - a_k x^4 / 4 + b_k x^2 / 2 + d_k x
a_true <- c(0.7, 1.5)
b_true <- c(-1.0, 0.5)
d_true <- c(-0.8, 1.0)

############################################################
# 1. Stationary distribution
############################################################

stationary_dist <- function(A) {
  K <- nrow(A)

  M <- t(A) - diag(K)
  M[K, ] <- rep(1, K)

  b <- c(rep(0, K - 1), 1)
  rho <- solve(M, b)

  rho <- pmax(rho, 0)
  rho / sum(rho)
}

rho_true <- stationary_dist(A_true)

############################################################
# 2. Unnormalized quartic observation model
############################################################

log_tilde_p_quartic <- function(x, log_a, b, d) {
  a <- exp(log_a)
  -0.25 * a * x^4 + 0.5 * b * x^2 + d * x
}

score_quartic <- function(x, log_a, b, d, c) {
  log_tilde_p_quartic(x, log_a, b, d) + c
}

############################################################
# 3. Normalizing constants
#    Diagnostic only. These are not used for estimation.
############################################################

compute_logZ_quartic <- function(log_a, b, d) {
  f <- function(x) {
    exp(log_tilde_p_quartic(x, log_a, b, d))
  }

  val <- integrate(
    f,
    lower = -Inf,
    upper = Inf,
    rel.tol = 1e-10,
    abs.tol = 1e-12,
    subdivisions = 2000
  )$value

  log(val)
}

log_a_true <- log(a_true)

logZ_true <- sapply(1:K, function(k) {
  compute_logZ_quartic(
    log_a = log_a_true[k],
    b = b_true[k],
    d = d_true[k]
  )
})

c_true <- -logZ_true

############################################################
# 4. Sampling from the true quartic distributions
#    One-dimensional inverse-CDF approximation on a grid.
############################################################

make_quartic_sampler <- function(
  log_a,
  b,
  d,
  lower = -6,
  upper = 6,
  grid_size = 50000
) {
  grid <- seq(lower, upper, length.out = grid_size)

  logw <- log_tilde_p_quartic(
    x = grid,
    log_a = log_a,
    b = b,
    d = d
  )

  w <- exp(logw - max(logw))
  w <- w / sum(w)
  cdf <- cumsum(w)

  function(n) {
    u <- runif(n)
    idx <- findInterval(u, cdf) + 1
    idx[idx < 1] <- 1
    idx[idx > grid_size] <- grid_size
    grid[idx]
  }
}

sampler_list <- vector("list", K)

for (k in 1:K) {
  sampler_list[[k]] <- make_quartic_sampler(
    log_a = log_a_true[k],
    b = b_true[k],
    d = d_true[k]
  )
}

############################################################
# 5. HMM data generation
############################################################

simulate_hmm_quartic <- function(T_len, A, rho, sampler_list) {
  K <- length(rho)

  z <- integer(T_len)
  x <- numeric(T_len)

  z[1] <- sample(1:K, size = 1, prob = rho)
  x[1] <- sampler_list[[z[1]]](1)

  for (t in 2:T_len) {
    z[t] <- sample(1:K, size = 1, prob = A[z[t - 1], ])
    x[t] <- sampler_list[[z[t]]](1)
  }

  list(x = x, z = z)
}

dat <- simulate_hmm_quartic(
  T_len = T_len,
  A = A_true,
  rho = rho_true,
  sampler_list = sampler_list
)

x <- dat$x
z_true <- dat$z

############################################################
# 6. Filtering and smoothing
############################################################

filter_smooth_quartic <- function(x, A, rho, log_a, b, d, c) {
  T_len <- length(x)
  K <- length(rho)

  log_bmat <- matrix(0, nrow = T_len, ncol = K)

  for (k in 1:K) {
    log_bmat[, k] <- score_quartic(
      x = x,
      log_a = log_a[k],
      b = b[k],
      d = d[k],
      c = c[k]
    )
  }

  log_b_shift <- log_bmat - apply(log_bmat, 1, max)
  bmat <- exp(log_b_shift)

  alpha <- matrix(0, nrow = T_len, ncol = K)

  alpha[1, ] <- rho * bmat[1, ]
  alpha[1, ] <- alpha[1, ] / sum(alpha[1, ])

  for (t in 2:T_len) {
    pred <- as.numeric(alpha[t - 1, ] %*% A)
    alpha[t, ] <- bmat[t, ] * pred
    alpha[t, ] <- alpha[t, ] / sum(alpha[t, ])
  }

  beta <- matrix(0, nrow = T_len, ncol = K)
  beta[T_len, ] <- rep(1, K)

  for (t in (T_len - 1):1) {
    for (j in 1:K) {
      beta[t, j] <- sum(
        A[j, ] * bmat[t + 1, ] * beta[t + 1, ]
      )
    }
    beta[t, ] <- beta[t, ] / sum(beta[t, ])
  }

  gamma <- alpha * beta
  gamma <- gamma / rowSums(gamma)

  list(
    gamma = gamma,
    alpha = alpha,
    beta = beta,
    log_b = log_bmat
  )
}

############################################################
# 7. Weighted NCE objective
############################################################

weighted_nce_objective_quartic <- function(
  par,
  x,
  gamma_k,
  noise_y,
  noise_density_fun
) {
  log_a <- par[1]
  b <- par[2]
  d <- par[3]
  c <- par[4]

  Nk <- sum(gamma_k)
  Mk <- length(noise_y)

  sx <- score_quartic(x, log_a, b, d, c)
  sy <- score_quartic(noise_y, log_a, b, d, c)

  nx <- noise_density_fun(x)
  ny <- noise_density_fun(noise_y)

  log_num_x <- log(Nk) + sx
  log_noise_x <- log(Mk) + log(nx)

  max_x <- pmax(log_num_x, log_noise_x)
  log_den_x <- max_x +
    log(exp(log_num_x - max_x) + exp(log_noise_x - max_x))

  log_D_x <- log_num_x - log_den_x

  log_num_y <- log(Nk) + sy
  log_noise_y <- log(Mk) + log(ny)

  max_y <- pmax(log_num_y, log_noise_y)
  log_den_y <- max_y +
    log(exp(log_num_y - max_y) + exp(log_noise_y - max_y))

  log_1m_D_y <- log_noise_y - log_den_y

  obj <- sum(gamma_k * log_D_x) + sum(log_1m_D_y)

  -obj
}

update_one_state_nce_quartic <- function(
  x,
  gamma_k,
  init_par,
  noise_y,
  noise_density_fun
) {
  fit <- optim(
    par = init_par,
    fn = weighted_nce_objective_quartic,
    x = x,
    gamma_k = gamma_k,
    noise_y = noise_y,
    noise_density_fun = noise_density_fun,
    method = "BFGS",
    control = list(maxit = 500, reltol = 1e-9)
  )

  fit$par
}

############################################################
# 8. Weighted NCE-HMM
#    A is known in this experiment.
############################################################

fit_weighted_nce_hmm_quartic <- function(
  x,
  A,
  rho,
  K,
  R = 50,
  noise_ratio = 10,
  damping = 0.5,
  tol = 1e-6
) {
  T_len <- length(x)

  km <- kmeans(x, centers = K, nstart = 30)
  cl <- km$cluster

  log_a <- rep(log(1), K)
  b <- rep(0, K)
  d <- numeric(K)
  c <- rep(0, K)

  for (k in 1:K) {
    xk <- x[cl == k]
    if (length(xk) < 10) {
      xk <- x
    }

    d[k] <- mean(xk)
  }

  ord <- order(d)
  log_a <- log_a[ord]
  b <- b[ord]
  d <- d[ord]
  c <- c[ord]

  noise_mean <- mean(x)
  noise_sd <- sd(x) * 2

  noise_density_fun <- function(u) {
    dnorm(u, mean = noise_mean, sd = noise_sd)
  }

  M_fixed <- max(500, round(noise_ratio * T_len / K))

  noise_list <- vector("list", K)

  for (k in 1:K) {
    noise_list[[k]] <- rnorm(
      M_fixed,
      mean = noise_mean,
      sd = noise_sd
    )
  }

  history <- data.frame()

  for (r in 1:R) {
    old_log_a <- log_a
    old_b <- b
    old_d <- d
    old_c <- c

    fit_state <- filter_smooth_quartic(
      x = x,
      A = A,
      rho = rho,
      log_a = log_a,
      b = b,
      d = d,
      c = c
    )

    gamma <- fit_state$gamma

    for (k in 1:K) {
      init_par <- c(log_a[k], b[k], d[k], c[k])

      new_par <- update_one_state_nce_quartic(
        x = x,
        gamma_k = gamma[, k],
        init_par = init_par,
        noise_y = noise_list[[k]],
        noise_density_fun = noise_density_fun
      )

      log_a[k] <- (1 - damping) * log_a[k] + damping * new_par[1]
      b[k] <- (1 - damping) * b[k] + damping * new_par[2]
      d[k] <- (1 - damping) * d[k] + damping * new_par[3]
      c[k] <- (1 - damping) * c[k] + damping * new_par[4]
    }

    ord <- order(d)

    log_a <- log_a[ord]
    b <- b[ord]
    d <- d[ord]
    c <- c[ord]

    diff <- sum(abs(log_a - old_log_a)) +
      sum(abs(b - old_b)) +
      sum(abs(d - old_d)) +
      sum(abs(c - old_c))

    history <- rbind(
      history,
      data.frame(
        iter = r,
        diff = diff,
        a1 = exp(log_a[1]),
        a2 = exp(log_a[2]),
        b1 = b[1],
        b2 = b[2],
        d1 = d[1],
        d2 = d[2],
        c1 = c[1],
        c2 = c[2]
      )
    )

    if (diff < tol) {
      break
    }
  }

  fit_final <- filter_smooth_quartic(
    x = x,
    A = A,
    rho = rho,
    log_a = log_a,
    b = b,
    d = d,
    c = c
  )

  list(
    log_a = log_a,
    a = exp(log_a),
    b = b,
    d = d,
    c = c,
    gamma = fit_final$gamma,
    history = history
  )
}

############################################################
# 9. Estimation
############################################################

fit <- fit_weighted_nce_hmm_quartic(
  x = x,
  A = A_true,
  rho = rho_true,
  K = K,
  R = 50,
  noise_ratio = 10,
  damping = 0.5
)

############################################################
# 10. Diagnostic -log Z(hat theta)
############################################################

logZ_hat <- sapply(1:K, function(k) {
  compute_logZ_quartic(
    log_a = fit$log_a[k],
    b = fit$b[k],
    d = fit$d[k]
  )
})

minus_logZ_hat <- -logZ_hat

############################################################
# 11. Oracle / Weighted NCE-HMM / Naive
############################################################

fit_oracle <- filter_smooth_quartic(
  x = x,
  A = A_true,
  rho = rho_true,
  log_a = log_a_true,
  b = b_true,
  d = d_true,
  c = c_true
)

gamma_oracle <- fit_oracle$gamma

fit_naive <- filter_smooth_quartic(
  x = x,
  A = A_true,
  rho = rho_true,
  log_a = fit$log_a,
  b = fit$b,
  d = fit$d,
  c = rep(0, K)
)

gamma_naive <- fit_naive$gamma

############################################################
# 12. Evaluation functions
############################################################

best_accuracy_2state <- function(z_hat, z_true) {
  acc1 <- mean(z_hat == z_true)
  acc2 <- mean((3 - z_hat) == z_true)
  max(acc1, acc2)
}

l1_gamma_distance <- function(gamma_hat, gamma_ref) {
  mean(rowSums(abs(gamma_hat - gamma_ref)))
}

############################################################
# 13. Results
############################################################

z_hat_oracle <- apply(gamma_oracle, 1, which.max)
z_hat_wnce <- apply(fit$gamma, 1, which.max)
z_hat_naive <- apply(gamma_naive, 1, which.max)

cat("True a:\n")
print(a_true)

cat("Estimated a:\n")
print(fit$a)

cat("True b:\n")
print(b_true)

cat("Estimated b:\n")
print(fit$b)

cat("True d:\n")
print(d_true)

cat("Estimated d:\n")
print(fit$d)

cat("True c = -log Z(theta*):\n")
print(c_true)

cat("Estimated c:\n")
print(fit$c)

cat("-log Z(hat theta):\n")
print(minus_logZ_hat)

cat("c comparison:\n")
print(
  rbind(
    c_true = c_true,
    c_hat = fit$c,
    minus_logZ_hat = minus_logZ_hat
  )
)

comparison <- data.frame(
  method = c(
    "Oracle",
    "Weighted NCE-HMM",
    "Naive"
  ),
  accuracy = c(
    best_accuracy_2state(z_hat_oracle, z_true),
    best_accuracy_2state(z_hat_wnce, z_true),
    best_accuracy_2state(z_hat_naive, z_true)
  ),
  gamma_L1_to_oracle = c(
    0,
    l1_gamma_distance(fit$gamma, gamma_oracle),
    l1_gamma_distance(gamma_naive, gamma_oracle)
  )
)

print(comparison)
print(fit$history)

############################################################
# 14. True and estimated density plots
############################################################

x_grid <- seq(
  min(x) - 1,
  max(x) + 1,
  length.out = 1000
)

true_density <- matrix(0, nrow = length(x_grid), ncol = K)
estimated_density <- matrix(0, nrow = length(x_grid), ncol = K)

for (k in 1:K) {
  true_density[, k] <- exp(
    score_quartic(
      x = x_grid,
      log_a = log_a_true[k],
      b = b_true[k],
      d = d_true[k],
      c = c_true[k]
    )
  )

  estimated_density[, k] <- exp(
    score_quartic(
      x = x_grid,
      log_a = fit$log_a[k],
      b = fit$b[k],
      d = fit$d[k],
      c = fit$c[k]
    )
  )
}

par(mfrow = c(2, 2))

plot(
  x,
  type = "l",
  main = "Observed time series",
  xlab = "t",
  ylab = expression(x[t])
)

plot(
  fit$gamma[, 1],
  type = "l",
  ylim = c(0, 1),
  main = "Smoothing probability: state 1",
  xlab = "t",
  ylab = expression(gamma[t](1))
)

plot(
  x_grid,
  true_density[, 1],
  type = "l",
  ylim = range(c(true_density[, 1], estimated_density[, 1])),
  main = "State 1 density",
  xlab = "x",
  ylab = "density"
)
lines(
  x_grid,
  estimated_density[, 1],
  lty = 2
)
legend(
  "topright",
  legend = c("true", "estimated"),
  lty = c(1, 2),
  bty = "n"
)

plot(
  x_grid,
  true_density[, 2],
  type = "l",
  ylim = range(c(true_density[, 2], estimated_density[, 2])),
  main = "State 2 density",
  xlab = "x",
  ylab = "density"
)
lines(
  x_grid,
  estimated_density[, 2],
  lty = 2
)
legend(
  "topright",
  legend = c("true", "estimated"),
  lty = c(1, 2),
  bty = "n"
)
