import math
import numpy as np
import pandas as pd
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GumbelCopula

def predict_median_time_from_survival_curves(surv_probs: np.ndarray,
                                             time_grid: np.ndarray) -> np.ndarray:
    n, m = surv_probs.shape
    med = np.full(n, float(time_grid[-1]), dtype=float)
    for i in range(n):
        idx = np.where(surv_probs[i] <= 0.5)[0]
        if idx.size > 0:
            med[i] = float(time_grid[idx[0]])
    return med

def convert_to_structured(T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def kendall_tau_to_theta(copula_name, k_tau):
    if copula_name == "clayton":
        return ClaytonCopula().theta_from_tau(k_tau)
    elif copula_name == "frank":
        return FrankCopula().theta_from_tau(k_tau)
    elif copula_name == "gumbel":
        return GumbelCopula().theta_from_tau(k_tau)
    else:
        raise NotImplementedError('Copula not implemented')
    
def theta_to_kendall_tau(copula_name, theta):
    if copula_name == "clayton":
        return ClaytonCopula().tau(theta)
    elif copula_name == "frank":
        return FrankCopula().tau(theta)
    elif copula_name == "gumbel":
        return GumbelCopula().tau(theta)
    else:
        raise NotImplementedError('Copula not implemented')
    
def train_test_split_df(df, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(train_frac * len(df))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

def lifelines_surv_to_matrix(surv_df_or_list):
    """
    Returns:
      time_grid: 1D np array of times (m,)
      S: np array shape (n, m)
    """
    if isinstance(surv_df_or_list, pd.DataFrame):
        # lifelines typical: index=time, columns=individuals
        time_grid = surv_df_or_list.index.values.astype(float)
        S = surv_df_or_list.T.values.astype(float)  # (n, m)
        return time_grid, S

    # if it's a list/iterator of callables or series, fallback:
    surv_list = list(surv_df_or_list)
    # each entry expected to be a pandas Series indexed by time
    time_grid = surv_list[0].index.values.astype(float)
    S = np.row_stack([s.values.astype(float) for s in surv_list])
    return time_grid, S

def make_time_bins(times, num_bins=None, use_quantiles=True, event=None):
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    return bins