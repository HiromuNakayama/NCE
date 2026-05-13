set.seed(123)

############################################################
# 0. Common functions from the Gaussian NCE-HMM experiment
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

simulate_hmm <- function(T_len, A, rho, mu, lambda) {
  K <- length(rho)

  z <- integer(T_len)
  x <- numeric(T_len)

  z[1] <- sample(1:K, size = 1, prob = rho)
  x[1] <- rnorm(
    1,
    mean = mu[z[1]],
    sd = 1 / sqrt(lambda[z[1]])
  )

  for (t in 2:T_len) {
    z[t] <- sample(1:K, size = 1, prob = A[z[t - 1], ])
    x[t] <- rnorm(
      1,
      mean = mu[z[t]],
      sd = 1 / sqrt(lambda[z[t]])
    )
  }

  list(x = x, z = z)
}

log_tilde_p <- function(x, mu, log_lambda) {
  lambda <- exp(log_lambda)
  -0.5 * lambda * (x - mu)^2
}

score_fun <- function(x, mu, log_lambda, c) {
  log_tilde_p(x, mu, log_lambda) + c
}

filter_smooth_with_xi <- function(x, A, rho, mu, log_lambda, c) {
  T_len <- length(x)
  K <- length(rho)

  log_b <- matrix(0, nrow = T_len, ncol = K)

  for (k in 1:K) {
    log_b[, k] <- score_fun(
      x = x,
      mu = mu[k],
      log_lambda = log_lambda[k],
      c = c[k]
    )
  }

  log_b_shift <- log_b - apply(log_b, 1, max)
  b <- exp(log_b_shift)

  alpha <- matrix(0, nrow = T_len, ncol = K)

  alpha[1, ] <- rho * b[1, ]
  alpha[1, ] <- alpha[1, ] / sum(alpha[1, ])

  for (t in 2:T_len) {
    pred <- as.numeric(alpha[t - 1, ] %*% A)
    alpha[t, ] <- b[t, ] * pred
    alpha[t, ] <- alpha[t, ] / sum(alpha[t, ])
  }

  beta <- matrix(0, nrow = T_len, ncol = K)
  beta[T_len, ] <- rep(1, K)

  for (t in (T_len - 1):1) {
    for (j in 1:K) {
      beta[t, j] <- sum(
        A[j, ] * b[t + 1, ] * beta[t + 1, ]
      )
    }
    beta[t, ] <- beta[t, ] / sum(beta[t, ])
  }

  gamma <- alpha * beta
  gamma <- gamma / rowSums(gamma)

  xi <- array(0, dim = c(T_len, K, K))

  for (t in 2:T_len) {
    numer <- matrix(0, nrow = K, ncol = K)

    for (j in 1:K) {
      for (k in 1:K) {
        numer[j, k] <-
          alpha[t - 1, j] *
          A[j, k] *
          b[t, k] *
          beta[t, k]
      }
    }

    xi[t, , ] <- numer / sum(numer)
  }

  list(
    gamma = gamma,
    xi = xi,
    alpha = alpha,
    beta = beta,
    log_b = log_b
  )
}

weighted_nce_objective <- function(
  par,
  x,
  gamma_k,
  noise_y,
  noise_density_fun
) {
  mu <- par[1]
  log_lambda <- par[2]
  c <- par[3]

  Nk <- sum(gamma_k)
  Mk <- length(noise_y)

  sx <- score_fun(x, mu, log_lambda, c)
  sy <- score_fun(noise_y, mu, log_lambda, c)

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

update_one_state_nce <- function(
  x,
  gamma_k,
  init_par,
  noise_y,
  noise_density_fun
) {
  fit <- optim(
    par = init_par,
    fn = weighted_nce_objective,
    x = x,
    gamma_k = gamma_k,
    noise_y = noise_y,
    noise_density_fun = noise_density_fun,
    method = "BFGS",
    control = list(maxit = 300, reltol = 1e-8)
  )

  fit$par
}

initialize_A_from_labels <- function(labels, K, pseudocount = 1) {
  counts <- matrix(pseudocount, nrow = K, ncol = K)

  for (t in 2:length(labels)) {
    counts[labels[t - 1], labels[t]] <-
      counts[labels[t - 1], labels[t]] + 1
  }

  counts / rowSums(counts)
}

fit_weighted_nce_hmm_unknown_A <- function(
  x,
  K,
  R = 50,
  noise_ratio = 5,
  damping = 0.7,
  tol = 1e-6
) {
  T_len <- length(x)

  km <- kmeans(x, centers = K, nstart = 30)
  cl <- km$cluster

  mu <- numeric(K)
  log_lambda <- numeric(K)
  c <- rep(0, K)

  for (k in 1:K) {
    xk <- x[cl == k]
    if (length(xk) < 5) {
      xk <- x
    }

    mu[k] <- mean(xk)
    log_lambda[k] <- log(1 / var(xk))
  }

  ord <- order(mu)
  mu <- mu[ord]
  log_lambda <- log_lambda[ord]
  c <- c[ord]

  new_label <- integer(K)
  new_label[ord] <- 1:K
  cl <- new_label[cl]

  A <- initialize_A_from_labels(cl, K)

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
    old_mu <- mu
    old_log_lambda <- log_lambda
    old_c <- c
    old_A <- A

    rho <- stationary_dist(A)

    fit_state <- filter_smooth_with_xi(
      x = x,
      A = A,
      rho = rho,
      mu = mu,
      log_lambda = log_lambda,
      c = c
    )

    gamma <- fit_state$gamma
    xi <- fit_state$xi

    A_new <- matrix(0, nrow = K, ncol = K)

    for (j in 1:K) {
      denom <- sum(gamma[1:(T_len - 1), j])

      for (k in 1:K) {
        numer <- sum(xi[2:T_len, j, k])
        A_new[j, k] <- numer / denom
      }
    }

    A_new <- A_new / rowSums(A_new)

    A <- (1 - damping) * A + damping * A_new
    A <- A / rowSums(A)

    for (k in 1:K) {
      init_par <- c(mu[k], log_lambda[k], c[k])

      new_par <- update_one_state_nce(
        x = x,
        gamma_k = gamma[, k],
        init_par = init_par,
        noise_y = noise_list[[k]],
        noise_density_fun = noise_density_fun
      )

      mu[k] <- (1 - damping) * mu[k] + damping * new_par[1]
      log_lambda[k] <-
        (1 - damping) * log_lambda[k] + damping * new_par[2]
      c[k] <- (1 - damping) * c[k] + damping * new_par[3]
    }

    ord <- order(mu)

    mu <- mu[ord]
    log_lambda <- log_lambda[ord]
    c <- c[ord]
    A <- A[ord, ord]

    diff <- sum(abs(mu - old_mu)) +
      sum(abs(log_lambda - old_log_lambda)) +
      sum(abs(c - old_c)) +
      sum(abs(A - old_A))

    rho_current <- stationary_dist(A)

    history <- rbind(
      history,
      data.frame(
        iter = r,
        diff = diff,
        mu1 = mu[1],
        mu2 = mu[2],
        lambda1 = exp(log_lambda[1]),
        lambda2 = exp(log_lambda[2]),
        c1 = c[1],
        c2 = c[2],
        A11 = A[1, 1],
        A12 = A[1, 2],
        A21 = A[2, 1],
        A22 = A[2, 2],
        rho1 = rho_current[1],
        rho2 = rho_current[2]
      )
    )

    if (diff < tol) {
      break
    }
  }

  rho_final <- stationary_dist(A)

  fit_final <- filter_smooth_with_xi(
    x = x,
    A = A,
    rho = rho_final,
    mu = mu,
    log_lambda = log_lambda,
    c = c
  )

  list(
    mu = mu,
    lambda = exp(log_lambda),
    log_lambda = log_lambda,
    c = c,
    A = A,
    rho = rho_final,
    gamma = fit_final$gamma,
    history = history
  )
}

best_accuracy_2state <- function(z_hat, z_true) {
  acc1 <- mean(z_hat == z_true)
  acc2 <- mean((3 - z_hat) == z_true)
  max(acc1, acc2)
}

############################################################
# Experiment 5: sample-size convergence experiment
############################################################

set.seed(123)

############################################################
# 1. Common settings
############################################################

parse_integer_grid <- function(value, default) {
  if (is.na(value) || value == "") {
    return(default)
  }

  as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
}

K <- 2
T_grid <- parse_integer_grid(
  Sys.getenv("NCE_SAMPLE_T_GRID", unset = ""),
  c(500, 1000, 2000, 5000, 10000)
)
B <- as.integer(Sys.getenv("NCE_SAMPLE_B", unset = "300"))
R_fit <- as.integer(Sys.getenv("NCE_SAMPLE_R", unset = "50"))

mu_true <- c(-1, 1)
lambda_true <- c(0.5, 3)

A_true <- matrix(
  c(0.95, 0.05,
    0.10, 0.90),
  nrow = 2,
  byrow = TRUE
)

rho_true <- stationary_dist(A_true)

c_true <- -0.5 * log(2 * pi) + 0.5 * log(lambda_true)

eta_true <- c(
  mu1 = mu_true[1],
  mu2 = mu_true[2],
  loglambda1 = log(lambda_true[1]),
  loglambda2 = log(lambda_true[2]),
  c1 = c_true[1],
  c2 = c_true[2],
  A12 = A_true[1, 2],
  A21 = A_true[2, 1]
)

############################################################
# 2. One replication
############################################################

run_one_sample_size_experiment <- function(
  T_len,
  K,
  A_true,
  rho_true,
  mu_true,
  lambda_true,
  c_true,
  eta_true,
  R = 50,
  noise_ratio = 5,
  damping = 0.7
) {
  dat <- simulate_hmm(
    T_len = T_len,
    A = A_true,
    rho = rho_true,
    mu = mu_true,
    lambda = lambda_true
  )

  x <- dat$x
  z_true <- dat$z

  fit <- fit_weighted_nce_hmm_unknown_A(
    x = x,
    K = K,
    R = R,
    noise_ratio = noise_ratio,
    damping = damping
  )

  mu_error <- sqrt(sum((fit$mu - mu_true)^2))
  lambda_error <- sqrt(sum((fit$lambda - lambda_true)^2))
  c_error <- sqrt(sum((fit$c - c_true)^2))
  A_error <- norm(fit$A - A_true, type = "F")

  z_hat <- apply(fit$gamma, 1, which.max)
  accuracy <- best_accuracy_2state(z_hat, z_true)

  eta_hat <- c(
    mu1 = fit$mu[1],
    mu2 = fit$mu[2],
    loglambda1 = log(fit$lambda[1]),
    loglambda2 = log(fit$lambda[2]),
    c1 = fit$c[1],
    c2 = fit$c[2],
    A12 = fit$A[1, 2],
    A21 = fit$A[2, 1]
  )

  scaled_error <- sqrt(T_len) * (eta_hat - eta_true)

  data.frame(
    T = T_len,
    mu_error = mu_error,
    lambda_error = lambda_error,
    c_error = c_error,
    A_error = A_error,
    accuracy = accuracy,
    mu1_hat = fit$mu[1],
    mu2_hat = fit$mu[2],
    lambda1_hat = fit$lambda[1],
    lambda2_hat = fit$lambda[2],
    c1_hat = fit$c[1],
    c2_hat = fit$c[2],
    A12_hat = fit$A[1, 2],
    A21_hat = fit$A[2, 1],
    sqrtT_mu1 = scaled_error["mu1"],
    sqrtT_mu2 = scaled_error["mu2"],
    sqrtT_loglambda1 = scaled_error["loglambda1"],
    sqrtT_loglambda2 = scaled_error["loglambda2"],
    sqrtT_c1 = scaled_error["c1"],
    sqrtT_c2 = scaled_error["c2"],
    sqrtT_A12 = scaled_error["A12"],
    sqrtT_A21 = scaled_error["A21"]
  )
}

############################################################
# 3. Run all experiments
############################################################

results_list <- vector("list", length(T_grid) * B)
idx <- 1

for (T_len in T_grid) {
  for (b in 1:B) {
    cat("T =", T_len, " replication =", b, "\n")

    results_list[[idx]] <- run_one_sample_size_experiment(
      T_len = T_len,
      K = K,
      A_true = A_true,
      rho_true = rho_true,
      mu_true = mu_true,
      lambda_true = lambda_true,
      c_true = c_true,
      eta_true = eta_true,
      R = R_fit,
      noise_ratio = 5,
      damping = 0.7
    )

    idx <- idx + 1
  }
}

results_sample_size <- do.call(rbind, results_list)

############################################################
# 4. Mean and SD summaries
############################################################

summary_mean <- aggregate(
  cbind(
    mu_error,
    lambda_error,
    c_error,
    A_error,
    accuracy
  ) ~ T,
  data = results_sample_size,
  FUN = mean
)

summary_sd <- aggregate(
  cbind(
    mu_error,
    lambda_error,
    c_error,
    A_error,
    accuracy
  ) ~ T,
  data = results_sample_size,
  FUN = sd
)

cat("Mean:\n")
print(summary_mean)

cat("SD:\n")
print(summary_sd)

############################################################
# 5. Readable summary table
############################################################

format_mean_sd <- function(mean_vec, sd_vec, digits = 4) {
  paste0(
    formatC(mean_vec, format = "f", digits = digits),
    " (",
    formatC(sd_vec, format = "f", digits = digits),
    ")"
  )
}

summary_table <- data.frame(
  T = summary_mean$T,
  mu_error = format_mean_sd(
    summary_mean$mu_error,
    summary_sd$mu_error
  ),
  lambda_error = format_mean_sd(
    summary_mean$lambda_error,
    summary_sd$lambda_error
  ),
  c_error = format_mean_sd(
    summary_mean$c_error,
    summary_sd$c_error
  ),
  A_error = format_mean_sd(
    summary_mean$A_error,
    summary_sd$A_error
  ),
  accuracy = format_mean_sd(
    summary_mean$accuracy,
    summary_sd$accuracy
  )
)

print(summary_table)

############################################################
# 6. Error decay plots
############################################################

par(mfrow = c(2, 2))

plot(
  summary_mean$T,
  summary_mean$mu_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(mu) - mu^"*" * "||"[2]),
  main = expression("Error of " * hat(mu))
)

plot(
  summary_mean$T,
  summary_mean$lambda_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(lambda) - lambda^"*" * "||"[2]),
  main = expression("Error of " * hat(lambda))
)

plot(
  summary_mean$T,
  summary_mean$c_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(c) - c^"*" * "||"[2]),
  main = expression("Error of " * hat(c))
)

plot(
  summary_mean$T,
  summary_mean$A_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(A) - A^"*" * "||"[F]),
  main = expression("Error of " * hat(A))
)

############################################################
# 7. Add 1/sqrt(T) reference lines
############################################################

add_reference_sqrt_rate <- function(T_vec, error_vec) {
  ref <- error_vec[1] * sqrt(T_vec[1] / T_vec)
  lines(T_vec, ref, lty = 2)
}

par(mfrow = c(2, 2))

plot(
  summary_mean$T,
  summary_mean$mu_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(mu) - mu^"*" * "||"[2]),
  main = expression("Error of " * hat(mu))
)
add_reference_sqrt_rate(summary_mean$T, summary_mean$mu_error)
legend(
  "topright",
  legend = c("observed", expression(T^{-1/2})),
  lty = c(1, 2),
  bty = "n"
)

plot(
  summary_mean$T,
  summary_mean$lambda_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(lambda) - lambda^"*" * "||"[2]),
  main = expression("Error of " * hat(lambda))
)
add_reference_sqrt_rate(summary_mean$T, summary_mean$lambda_error)
legend(
  "topright",
  legend = c("observed", expression(T^{-1/2})),
  lty = c(1, 2),
  bty = "n"
)

plot(
  summary_mean$T,
  summary_mean$c_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(c) - c^"*" * "||"[2]),
  main = expression("Error of " * hat(c))
)
add_reference_sqrt_rate(summary_mean$T, summary_mean$c_error)
legend(
  "topright",
  legend = c("observed", expression(T^{-1/2})),
  lty = c(1, 2),
  bty = "n"
)

plot(
  summary_mean$T,
  summary_mean$A_error,
  type = "b",
  log = "xy",
  xlab = "T",
  ylab = expression("Mean " * "||" * hat(A) - A^"*" * "||"[F]),
  main = expression("Error of " * hat(A))
)
add_reference_sqrt_rate(summary_mean$T, summary_mean$A_error)
legend(
  "topright",
  legend = c("observed", expression(T^{-1/2})),
  lty = c(1, 2),
  bty = "n"
)

############################################################
# 8. Asymptotic normality diagnostics
#    Use the largest sample size.
############################################################

results_large_T <- subset(
  results_sample_size,
  T == max(T_grid)
)

scaled_columns <- c(
  "sqrtT_mu1",
  "sqrtT_mu2",
  "sqrtT_loglambda1",
  "sqrtT_loglambda2",
  "sqrtT_c1",
  "sqrtT_c2",
  "sqrtT_A12",
  "sqrtT_A21"
)

############################################################
# 8.1 Histograms
############################################################

par(mfrow = c(2, 4))

for (v in scaled_columns) {
  hist(
    results_large_T[[v]],
    breaks = 20,
    probability = TRUE,
    main = v,
    xlab = ""
  )

  x_grid <- seq(
    min(results_large_T[[v]]),
    max(results_large_T[[v]]),
    length.out = 200
  )

  lines(
    x_grid,
    dnorm(
      x_grid,
      mean = mean(results_large_T[[v]]),
      sd = sd(results_large_T[[v]])
    ),
    lty = 2
  )
}

############################################################
# 8.2 QQ plots
############################################################

par(mfrow = c(2, 4))

for (v in scaled_columns) {
  qqnorm(
    results_large_T[[v]],
    main = v
  )
  qqline(results_large_T[[v]])
}
