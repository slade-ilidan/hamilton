import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_theta(y, *, K=2, m=1, const=True, noise=0.0001): # array, number of states, AR order, constant variance
    theta = {}
    for i in range(K): theta[i] = np.zeros(m + 1) + np.random.randn(m + 1) * noise # AR coefs plus constant
    theta['v'] = y.var() + np.random.randn() * noise if const else np.full(K, y.var()) + np.random.randn(K) * noise 
    theta['v'] = np.abs(theta['v'])
    theta['P'] = np.ones((K, K)) / K 
    theta['p'] = np.ones((m, K)) / K 
    theta['K'] = K
    theta['m'] = m
    theta['const'] = const
    return theta

def get_etas(y, theta):
    T, K, v, m = y.size, theta['K'], theta['v'], theta['m']
    mus = np.array([theta[state][0] for state in range(K)])
    coefs = np.array([theta[state][1:] for state in range(K)])
    lags = np.concatenate((np.full(m, y.mean()), y[:-1]))[:, None]
    product = np.array([np.convolve(lags[:, 0], coefs[i], mode='valid') for i in range(K)]).T
    etas = np.tile(y[:, None], (1, K)) - mus - product
    return (1 / np.sqrt(2 * np.pi * v)) * np.exp(-0.5 * etas**2 / v)

def get_filtered_and_predicted_regimes(etas, theta):
    T = len(etas)
    P = theta['P']
    p = theta['p']
    K = theta['K']
    filtered_regimes = np.zeros((T, K))
    predicted_regimes = np.zeros((T, K))
    predicted_regimes[0] = p[0]
    for t in range(T):
        prev_pred_etas = predicted_regimes[t] * etas[t]
        sum_prev_pred_etas = np.sum(prev_pred_etas)
        filtered_regimes[t] = prev_pred_etas / sum_prev_pred_etas
        if t == T - 1: break
        predicted_regimes[t + 1] = np.dot(P, filtered_regimes[t])
    return filtered_regimes, predicted_regimes

def get_smoothed_regimes(filtered_regimes, predicted_regimes, theta):
    T = len(filtered_regimes)
    PT = theta['P'].T
    smoothed_regimes = np.zeros_like(filtered_regimes)
    smoothed_regimes[-1] = filtered_regimes[-1]
    for t in range(T - 2, -1, -1):
        factor = np.dot(PT, smoothed_regimes[t + 1] / predicted_regimes[t + 1])
        smoothed_regimes[t] = filtered_regimes[t] * factor
    return smoothed_regimes

def get_transition_matrix(smoothed_regimes):
    prev_states = smoothed_regimes[:-1]
    next_states = smoothed_regimes[1:]
    joint_probabilities = prev_states.T @ next_states
    row_sums = prev_states.sum(axis=0)
    row_sums[row_sums == 0] = 1e-8
    transition_matrix = joint_probabilities / row_sums[:, None]
    return transition_matrix

def get_betas_mle(y, smoothed_regimes, theta):
    T = len(y)
    K = theta['K']
    m = theta['m']
    betas_mle = np.zeros((K, m + 1))
    Z = np.stack([np.concatenate(([1], y[t - m:t][::-1])) for t in range(m, T)])
    Y = y[m:T]
    for state in range(K):
        weights = np.sqrt(smoothed_regimes[m:T, state]) 
        Z_weighted = Z * weights[:, None]
        Y_weighted = Y * weights
        sum_zzT = Z_weighted.T @ Z_weighted
        sum_zy = Z_weighted.T @ Y_weighted
        betas_mle[state] = np.linalg.solve(sum_zzT, sum_zy)
    return betas_mle

def get_variance_mle(y, smoothed_regimes, betas_mle, theta):
    T = len(y)
    K = theta['K']
    m = theta['m']
    const = theta['const']
    Z = np.stack([np.concatenate(([1], y[t - m:t][::-1])) for t in range(m, T)])
    Y = y[m:T]
    weights = smoothed_regimes[m:T]
    Y_hat = Z @ betas_mle.T
    residuals_sq = (Y[:, None] - Y_hat) ** 2
    weighted_sq_residuals = np.sum(residuals_sq * weights, axis=0)
    if const:
        return np.sum(weighted_sq_residuals) / (T - m)
    else:
        state_weights_sum = np.sum(weights, axis=0)
        return weighted_sq_residuals / state_weights_sum

def get_log_likelihood(predicted_regimes, etas):
    return np.sum(np.log(np.sum(predicted_regimes * etas, axis=1)))

def get_aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood

def get_bic(log_likelihood, num_params, n):
    return np.log(n) * num_params - 2 * log_likelihood

def get_summary(y, theta, verbose=True):
    etas = get_etas(y, theta)
    f_regimes, p_regimes = get_filtered_and_predicted_regimes(etas, theta)
    ll = get_log_likelihood(p_regimes, etas)
    num_params = (theta['m'] + 1) * theta['K'] + theta['v'].size + theta['p'].size + theta['P'].size
    aic = get_aic(ll, num_params)
    bic = get_bic(ll, num_params, y.size)
    if verbose: print(f'LL={ll:.2f} AIC={aic:.2f} BIC={bic:.2f}')
    return {'ll': ll, 'aic': aic, 'bic': bic}

def em_optimize(y, theta, epsilon=1e-6):
    flatten_theta = lambda theta, keys: np.concatenate([theta[k].ravel() for k in keys])
    keys = list(range(theta['K'])) + ['v', 'p', 'P']
    prev_theta_flat = flatten_theta(theta, keys)
    iteration = 0
    while True:
        iteration += 1
        etas = get_etas(y, theta)
        f_regimes, p_regimes = get_filtered_and_predicted_regimes(etas, theta)
        s_regimes = get_smoothed_regimes(f_regimes, p_regimes, theta)

        betas_mle = get_betas_mle(y, s_regimes, theta)
        variance_mle = get_variance_mle(y, s_regimes, betas_mle, theta)

        K = theta['K']
        for state in range(K): theta[state] = betas_mle[state]
        theta['v'] = variance_mle
        theta['P'] = get_transition_matrix(s_regimes)
        theta['p'] = s_regimes[:theta['m']]

        theta_flat = flatten_theta(theta, keys)
        total_diff = np.linalg.norm(theta_flat - prev_theta_flat)
        print(f'#{iteration} total_diff {total_diff}')
        prev_theta_flat = theta_flat

        if total_diff < epsilon:
            break
    return theta

def get_conditional_forecasts(y, theta):
    T = y.size
    K = theta['K']
    m = theta['m']
    mus = np.array([theta[state][0] for state in range(K)])
    coefs = np.array([theta[state][1:][::-1] for state in range(K)])
    forecasts = np.full((T, K), np.nan)
    lags = np.array([y[i:i + m] for i in range(y.size - m)])
    forecasts = (lags @ coefs.T) + mus
    return np.vstack([np.full((m, K), np.nan), forecasts])

def get_forecasts(y, predicted_regimes, theta):
    T, K = predicted_regimes.shape
    conditional_forecasts = get_conditional_forecasts(y, theta)
    forecasts = np.sum(conditional_forecasts * predicted_regimes, axis=1)
    return forecasts

def plot_regime(c, regimes, regime, start=0, end=500):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(c[start:end], 'green')
    ax2.plot(regimes[start:end, regime], 'blue')
    plt.show()

def plot_regimes(c, regimes, start=0, end=500):
    K = regimes.shape[1]
    fig, axes = plt.subplots(K, 1, sharex=True)
    for i in range(K):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        ax1.plot(regimes[start:end, i], color='green')
        ax2.plot(c[start:end], color='blue')
    plt.show()

def plot_conditional_forecasts(r, cfcs, start=0, end=500):
    K = cfcs.shape[1]
    fig, axes = plt.subplots(K + 1, 1, sharex=True)
    axes[0].plot(r[start:end], color='blue')
    for i in range(K):
        axes[i + 1].plot(cfcs[start:end, i], color='green')
    plt.show()

def plot_forecasts(r, fcs, start=0, end=500):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(r[start:end], 'green')
    ax2.plot(fcs[start:end], 'blue')
    plt.show()

def save_theta(theta):
    K = theta['K']
    m = theta['m']
    const = 'T' if theta['const'] else 'F'
    path = f'theta_K={K}_m={m}_const={const}.pkl'
    with open(path, 'wb') as f: pickle.dump(theta, f)
    print(f'hamilton: {path} saved')

def load_theta(path):
    with open(path, 'rb') as f: theta = pickle.load(f)
    return theta
