import numpy as np
import pandas as pd
import torch
from pycop import simulation

from utility import convert_to_structured, kendall_tau_to_theta

class DGP_Weibull_linear:
    def __init__(self, n_features, alpha: float, gamma: float, use_x=True,
                 device="cpu", dtype=torch.float64, coeff=None, generator=None):
        self.alpha = torch.tensor([alpha], device=device).type(dtype)
        self.gamma = torch.tensor([gamma], device=device).type(dtype)
        self.use_x = use_x
        self.device = device
        self.dtype = dtype

        if coeff is not None:
            self.coeff = coeff.to(device=device, dtype=dtype)
        else:
            if generator is None:
                self.coeff = 2 * torch.rand((n_features,), device=device).type(dtype) - 1
            else:
                self.coeff = 2 * torch.rand((n_features,), generator=generator, device=device, dtype=dtype) - 1

    def PDF(self, t, x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self, t, x):
        return 1 - self.survival(t, x)
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        linear_term = torch.matmul(x, self.coeff) if self.use_x else 0.0
        return ((self.gamma / self.alpha) * ((t / self.alpha) ** (self.gamma - 1))) * torch.exp(linear_term)

    def cum_hazard(self, t, x):
        linear_term = torch.matmul(x, self.coeff) if self.use_x else 0.0
        return ((t / self.alpha) ** self.gamma) * torch.exp(linear_term)

    def parameters(self):
        return [self.alpha, self.gamma, self.coeff]
    
    def rvs(self, x, u):
        zero = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        linear_term = torch.matmul(x, self.coeff) if self.use_x else zero
        survival_term = -torch.log(u) / torch.exp(linear_term)
        result = (survival_term ** (1 / self.gamma)) * self.alpha
        return result.detach().cpu().numpy()
    
class SingleEventSyntheticDataLoader():
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  device='cpu', dtype=torch.float64):
        """
        This method generates synthetic data for single event (and censoring)
        DGP1: Data generation process for event
        DGP2: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
        dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
            
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uv = torch.stack([u, v], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v = simulation.simu_archimedean(copula_name, 2, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            uv = torch.cat([u, v], axis=1)
        
        t1_times = dgp1.rvs(X, uv[:,0].to(device))
        t2_times = dgp2.rvs(X, uv[:,1].to(device))
        
        observed_times = np.minimum(t1_times, t2_times)
        event_indicators = np.array((t2_times < t1_times), dtype=np.int32)
        
        self.true_censor_times = t1_times
        self.true_event_times = t2_times
    
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y = convert_to_structured(observed_times, event_indicators)
        self.dgps = [dgp1, dgp2]
        
        return self