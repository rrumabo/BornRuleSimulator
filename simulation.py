import numpy as np
from numpy.fft import fft, ifft, fftfreq

# --- Core Simulation Functions ---

def create_bimodal_superposition(x, x1, x2, sigma, a=1.0, b=1.0, phase_shift=0.0):
    """
    Create a bimodal superposition of two Gaussians.
    Each Gaussian is normalized (using np.trapezoid) before weighting.
    """
    phi1 = np.exp(-((x - x1)**2) / (2 * sigma**2))
    phi2 = np.exp(-((x - x2)**2) / (2 * sigma**2)) * np.exp(1j * phase_shift)
    phi1 /= np.sqrt(np.trapezoid(np.abs(phi1)**2, x))
    phi2 /= np.sqrt(np.trapezoid(np.abs(phi2)**2, x))
    psi = a * phi1 + b * phi2
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    return psi

def perturb_wavefunction(psi, x, noise_strength=0.05):
    """
    Apply phase noise to the wavefunction psi.
    """
    phase_noise = np.exp(1j * noise_strength * np.random.randn(len(x)))
    return psi * phase_noise / np.sqrt(np.trapezoid(np.abs(psi)**2, x))

def compute_kinetic(psi, k):
    """
    Compute the kinetic term using the FFT.
    """
    return -0.5 * ifft(k**2 * fft(psi))

def compute_entropy(psi, x):
    """
    Compute the entropy S and probability density rho.
    """
    rho = np.abs(psi)**2
    log_rho = np.log(np.maximum(rho, 1e-12))
    S = -np.trapezoid(rho * log_rho, x)
    return S, rho

def compute_mu(psi, kinetic, ent_force, x):
    """
    Compute the chemical potential (mu) by integrating the product
    of psi* and (kinetic + ent_force).
    """
    integrand = np.real(np.conjugate(psi) * (kinetic + ent_force))
    return np.trapezoid(integrand, x)

def evolve_wavefunction(psi, config, x, k):
    """
    Evolve the wavefunction psi using a kinetic/entropy scheme.
    The evolution is performed over config['num_steps'] time steps.
    For publication figures, consider increasing config['num_steps'].
    """
    num_steps = config['num_steps']
    d_tau = config['d_tau']
    T_param = config['T_param']
    alpha = config['alpha']
    imaginary_time = config.get('imaginary_time', True)
    
    for _ in range(num_steps):
        S, rho = compute_entropy(psi, x)
        kinetic = compute_kinetic(psi, k)
        log_rho = np.log(np.maximum(rho, 1e-12))
        ent_force = alpha * T_param * (log_rho + 1) * psi
        mu = compute_mu(psi, kinetic, ent_force, x)
        if imaginary_time:
            psi -= d_tau * (kinetic + ent_force - mu * psi)
        else:
            psi -= 1j * d_tau * (kinetic + ent_force - mu * psi)
        psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    return psi

def identify_attractor(psi, x):
    """
    Determines the attractor basin based on the expectation value ⟨x⟩.
    If x_mean < -0.2, returns 'left'; if x_mean > 0.2, returns 'right';
    otherwise returns 'uncertain'.
    Returns:
      (attractor, x_mean)
    """
    rho = np.abs(psi)**2
    x_mean = np.trapezoid(x * rho, x)
    if x_mean < -0.2:
        return 'left', x_mean
    elif x_mean > 0.2:
        return 'right', x_mean
    else:
        return 'uncertain', x_mean

# --- Multiprocessing Helpers ---

def simulate_single_perturbation(args):
    """
    Helper function for a single perturbation trial.
    Args is a tuple: (psi_base, config, x, k)
    Returns: (attractor, x_mean)
    """
    psi_base, config, x, k = args
    psi_perturbed = perturb_wavefunction(psi_base, x)
    psi_final = evolve_wavefunction(psi_perturbed, config, x, k)
    attractor, x_mean = identify_attractor(psi_final, x)
    return (attractor, x_mean)

def multi_test_sweep_mp(config, a_values, b_values, num_perturbations=1000):
    """
    Multiprocessing version of the multi-test sweep.
    For each (a, b) pair (of pre-normalized values), the pair is normalized and 
    the inner perturbation loop is parallelized over available CPU cores.
    
    Returns:
      - results_grid: aggregated attractor counts by (a_norm, b_norm)
      - raw_data_grid: list of raw ⟨x⟩ values for each configuration.
    """
    from multiprocessing import Pool  # Import Pool within the function
    results_grid = {}
    raw_data_grid = {}
    
    for a in a_values:
        for b in b_values:
            norm = np.sqrt(a**2 + b**2)
            a_norm, b_norm = a / norm, b / norm
            L = config['L']
            N = config['N']
            dx = L / N
            x = np.linspace(-L/2, L/2, N)
            k = fftfreq(N, d=dx) * 2 * np.pi
            
            psi_base = create_bimodal_superposition(
                x, x1=-6.0, x2=6.0, sigma=1.0,
                a=a_norm, b=b_norm, phase_shift=np.pi)
            args_list = [(psi_base, config, x, k) for _ in range(num_perturbations)]
            
            with Pool() as pool:
                results_list = pool.map(simulate_single_perturbation, args_list)
            
            results = {'left': 0, 'right': 0, 'uncertain': 0}
            raw_means = []
            for attractor, x_mean in results_list:
                raw_means.append(x_mean)
                results[attractor] += 1
                
            if results['uncertain'] > 0.05 * num_perturbations:
                print(f"Warning: {results['uncertain']} uncertain cases at a={a_norm}, b={b_norm}")
            if results['uncertain'] < 0.05 * num_perturbations:
                results.pop('uncertain')
                
            results_grid[(a_norm, b_norm)] = results
            raw_data_grid[(a_norm, b_norm)] = raw_means
            
    return results_grid, raw_data_grid
