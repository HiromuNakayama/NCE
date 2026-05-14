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

mu_true <- c(-1, 1)
lambda_true <- c(0.5, 3)

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

c_true <- -0.5 * log(2 * pi) + 0.5 * log(lambda_true)

############################################################
# 2. Data generation
############################################################

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

dat <- simulate_hmm(
  T_len = T_len,
  A = A_true,
  rho = rho_true,
  mu = mu_true,
  lambda = lambda_true
)

x <- dat$x
z_true <- dat$z

############################################################
# 3. Observation score
############################################################

log_tilde_p <- function(x, mu, log_lambda) {
  lambda <- exp(log_lambda)
  -0.5 * lambda * (x - mu)^2
}

score_fun <- function(x, mu, log_lambda, c) {
  log_tilde_p(x, mu, log_lambda) + c
}

############################################################
# 4. Filtering and smoothing
#    Returns gamma_t(k) and xi_t(j, k).
############################################################

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

  # Subtract a common constant at each time for numerical stability.
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

############################################################
# 5. Gaussian HMM observed log-likelihood
#    Diagnostic only. This is not used for estimation.
############################################################

observed_loglik_gaussian_hmm <- function(x, A, mu, lambda) {
  K <- length(mu)
  T_len <- length(x)

  rho <- stationary_dist(A)

  log_b <- matrix(0, nrow = T_len, ncol = K)

  for (k in 1:K) {
    log_b[, k] <-
      0.5 * log(lambda[k]) -
      0.5 * log(2 * pi) -
      0.5 * lambda[k] * (x - mu[k])^2
  }

  shift_1 <- max(log_b[1, ])
  alpha <- rho * exp(log_b[1, ] - shift_1)
  scale_1 <- sum(alpha)
  alpha <- alpha / scale_1

  loglik <- shift_1 + log(scale_1)

  for (t in 2:T_len) {
    pred <- as.numeric(alpha %*% A)
    shift_t <- max(log_b[t, ])
    alpha <- pred * exp(log_b[t, ] - shift_t)
    scale_t <- sum(alpha)
    alpha <- alpha / scale_t
    loglik <- loglik + shift_t + log(scale_t)
  }

  loglik
}

############################################################
# 6. Weighted NCE
############################################################

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

############################################################
# 7. Weighted NCE-HMM with unknown A
#    rho is recalculated as the stationary distribution of A.
############################################################

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

    loglik_before <- observed_loglik_gaussian_hmm(
      x = x,
      A = A,
      mu = mu,
      lambda = exp(log_lambda)
    )

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

    loglik_after_A <- observed_loglik_gaussian_hmm(
      x = x,
      A = A,
      mu = old_mu,
      lambda = exp(old_log_lambda)
    )

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

    loglik_after_NCE <- observed_loglik_gaussian_hmm(
      x = x,
      A = A,
      mu = mu,
      lambda = exp(log_lambda)
    )

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
        rho2 = rho_current[2],
        loglik_before = loglik_before,
        loglik_after_A = loglik_after_A,
        loglik_after_NCE = loglik_after_NCE,
        delta_loglik_A = loglik_after_A - loglik_before,
        delta_loglik_NCE = loglik_after_NCE - loglik_after_A,
        delta_loglik_total = loglik_after_NCE - loglik_before
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

############################################################
# 8. Estimation with multiple initial values
############################################################

# Number of runs with different initializations
num_runs <- 5

fit_list <- vector("list", num_runs)

for (run in 1:num_runs) {
  cat("Run", run, "of", num_runs, "\n")
  
  fit_list[[run]] <- fit_weighted_nce_hmm_unknown_A(
    x = x,
    K = K,
    R = 50,
    noise_ratio = 5,
    damping = 0.7
  )
}

# Use the first fit as the main result for comparison
fit <- fit_list[[1]]

############################################################
# 9. Results from first run
############################################################

cat("True mu:\n")
print(mu_true)

cat("Estimated mu (Run 1):\n")
print(fit$mu)

cat("True lambda:\n")
print(lambda_true)

cat("Estimated lambda (Run 1):\n")
print(fit$lambda)

cat("True c = -log Z:\n")
print(c_true)

cat("Estimated c (Run 1):\n")
print(fit$c)

cat("True A:\n")
print(A_true)

cat("Estimated A (Run 1):\n")
print(fit$A)

cat("True stationary rho:\n")
print(rho_true)

cat("Estimated stationary rho (Run 1):\n")
print(fit$rho)

z_hat <- apply(fit$gamma, 1, which.max)
acc <- mean(z_hat == z_true)

cat("State classification accuracy (Run 1):\n")
print(acc)

print(fit$history)

############################################################
# 9.5 Comparison of multiple runs
############################################################

# Helper functions for comparison
best_accuracy_2state <- function(z_hat, z_true) {
  acc1 <- mean(z_hat == z_true)
  acc2 <- mean((3 - z_hat) == z_true)
  max(acc1, acc2)
}

l1_gamma_distance <- function(gamma_hat, gamma_ref) {
  mean(rowSums(abs(gamma_hat - gamma_ref)))
}

cat("\n\n========== CONVERGENCE ANALYSIS: Multiple Initial Values ==========\n\n")

# Extract estimated parameters from each run
mu_estimates <- matrix(NA, nrow = num_runs, ncol = K)
lambda_estimates <- matrix(NA, nrow = num_runs, ncol = K)
c_estimates <- matrix(NA, nrow = num_runs, ncol = K)
accuracy_list <- numeric(num_runs)

for (run in 1:num_runs) {
  mu_estimates[run, ] <- fit_list[[run]]$mu
  lambda_estimates[run, ] <- fit_list[[run]]$lambda
  c_estimates[run, ] <- fit_list[[run]]$c
  
  z_hat_run <- apply(fit_list[[run]]$gamma, 1, which.max)
  accuracy_list[run] <- best_accuracy_2state(z_hat_run, z_true)
}

cat("Estimated mu across runs:\n")
print(mu_estimates)
cat("\n")

cat("Estimated lambda across runs:\n")
print(lambda_estimates)
cat("\n")

cat("Estimated c across runs:\n")
print(c_estimates)
cat("\n")

cat("Classification accuracy across runs:\n")
print(accuracy_list)
cat("\n")

# Calculate convergence metrics
cat("Standard deviations of mu estimates across runs:\n")
print(apply(mu_estimates, 2, sd))
cat("\n")

cat("Standard deviations of lambda estimates across runs:\n")
print(apply(lambda_estimates, 2, sd))
cat("\n")

cat("Standard deviations of c estimates across runs:\n")
print(apply(c_estimates, 2, sd))
cat("\n")

# Check if all runs converge to similar values
mu_cv <- apply(mu_estimates, 2, function(x) sd(x) / mean(abs(x)))
lambda_cv <- apply(lambda_estimates, 2, function(x) sd(x) / mean(abs(x)))
c_cv <- apply(c_estimates, 2, function(x) sd(x) / (mean(abs(x)) + 1e-10))

cat("Coefficient of variation for mu:\n")
print(mu_cv)
cat("\n")

cat("Coefficient of variation for lambda:\n")
print(lambda_cv)
cat("\n")

cat("Coefficient of variation for c:\n")
print(c_cv)
cat("\n")

if (all(mu_cv < 0.1) && all(lambda_cv < 0.1) && all(c_cv < 0.1)) {
  cat("✓ All runs converged to similar values (CV < 0.1)\n")
} else {
  cat("✗ Some parameters show variation across runs (CV ≥ 0.1)\n")
}

cat("\n========== END OF CONVERGENCE ANALYSIS ==========\n\n")

############################################################
# 11. Oracle / Weighted NCE / Naive comparison
############################################################

fit_oracle <- filter_smooth_with_xi(
  x = x,
  A = A_true,
  rho = rho_true,
  mu = mu_true,
  log_lambda = log(lambda_true),
  c = c_true
)

gamma_oracle <- fit_oracle$gamma
z_hat_oracle <- apply(gamma_oracle, 1, which.max)

gamma_wnce <- fit$gamma
z_hat_wnce <- apply(gamma_wnce, 1, which.max)

fit_naive <- filter_smooth_with_xi(
  x = x,
  A = fit$A,
  rho = fit$rho,
  mu = fit$mu,
  log_lambda = log(fit$lambda),
  c = rep(0, K)
)

gamma_naive <- fit_naive$gamma
z_hat_naive <- apply(gamma_naive, 1, which.max)

comparison <- data.frame(
  method = c(
    "Oracle",
    "Weighted NCE",
    "Naive: estimated A, estimated theta, c=0"
  ),
  accuracy = c(
    best_accuracy_2state(z_hat_oracle, z_true),
    best_accuracy_2state(z_hat_wnce, z_true),
    best_accuracy_2state(z_hat_naive, z_true)
  ),
  gamma_L1_to_oracle = c(
    0,
    l1_gamma_distance(gamma_wnce, gamma_oracle),
    l1_gamma_distance(gamma_naive, gamma_oracle)
  )
)

print(comparison)

############################################################
# 11. Visualization
############################################################

par(mfrow = c(2, 2))

plot(
  x,
  type = "l",
  main = "Observed time series",
  xlab = "t",
  ylab = expression(x[t])
)

plot(
  z_true,
  type = "s",
  main = "True hidden states",
  xlab = "t",
  ylab = expression(z[t])
)

plot(
  fit$gamma[, 1],
  type = "l",
  ylim = c(0, 1),
  main = "Estimated smoothing probability: state 1",
  xlab = "t",
  ylab = expression(gamma[t](1))
)

plot(
  fit$history$iter,
  fit$history$diff,
  type = "b",
  main = "Convergence",
  xlab = "iteration",
  ylab = "parameter change"
)
