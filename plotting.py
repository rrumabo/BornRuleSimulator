import matplotlib.pyplot as plt
import csv
import os
from scipy.stats import binom

# Setup results directories
results_dir = os.path.expanduser("~/Desktop/born_rule_results/")
raw_data_dir = os.path.join(results_dir, "raw_data")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(raw_data_dir, exist_ok=True)

def plot_sweep_results(results_grid, meta=""):
    """
    Generates a Born-vs-Measured scatter plot.
    Computes Born prediction and plots measured left frequency with 95% binomial CI error bars.
    """
    a_vals, b_vals, measured = [], [], []
    lower_errors, upper_errors = [], []
    for (a_norm, b_norm), results in results_grid.items():
        a_vals.append(a_norm)
        b_vals.append(b_norm)
        total = results['left'] + results.get('right', 0)
        left_freq = results['left'] / total
        (lower, upper) = binom.interval(0.95, total, left_freq)
        lower_rate = lower / total
        upper_rate = upper / total
        lower_errors.append(left_freq - lower_rate)
        upper_errors.append(upper_rate - left_freq)
        measured.append(left_freq)
    a_vals = np.array(a_vals)
    measured = np.array(measured)
    lower_errors = np.array(lower_errors)
    upper_errors = np.array(upper_errors)
    born_predicted = a_vals**2 / (a_vals**2 + (1 - a_vals**2))  # since a_norm^2 + b_norm^2 = 1
    plt.errorbar(born_predicted, measured,
                 yerr=[lower_errors, upper_errors],
                 fmt='o', color='blue', ecolor='gray', capsize=3)
    plt.xlabel("Born Prediction (a² / [a² + b²])")
    plt.ylabel("Measured Left Frequency")
    plt.title("Diagonal Consistency with Born Rule")
    plt.legend(["Samples"])
    filename = os.path.join(results_dir, f"born_rule_sweep_{meta}.png")
    plt.savefig(filename)
    plt.show()

def save_results_to_csv(results_grid, meta=""):
    """
    Saves aggregated results to a CSV file.
    Filename includes metadata for easier traceability.
    """
    filepath = os.path.join(results_dir, f"born_rule_sweep_{meta}.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["a_norm", "b_norm", "Left Count", "Right Count",
                         "Relative Left Frequency", "Relative Right Frequency"])
        for (a_norm, b_norm), results in results_grid.items():
            total = results['left'] + results.get('right', 0)
            writer.writerow([a_norm, b_norm, results['left'], results.get('right', 0),
                             results['left'] / total, results.get('right', 0) / total])
    print(f"Results saved to {filepath}")

def export_raw_data(raw_data_grid, meta=""):
    """
    Exports raw ⟨x⟩ data for each configuration into separate CSV files.
    """
    for (a_norm, b_norm), x_means in raw_data_grid.items():
        fname = f"raw_data_a_{a_norm:.3f}_b_{b_norm:.3f}_{meta}.csv"
        fpath = os.path.join(raw_data_dir, fname)
        with open(fpath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x_mean"])
            for val in x_means:
                writer.writerow([val])
    print(f"Raw data exported to folder: {raw_data_dir}")

def sensitivity_study(config, a, b, noise_strengths, num_perturbations=100, meta=""):
    """
    For a given (a, b) configuration (pre-normalization) runs simulations over a range of noise strengths.
    Plots measured left frequency (with 95% CI error bars) versus noise strength.
    """
    from simulation import create_bimodal_superposition, perturb_wavefunction, evolve_wavefunction, identify_attractor
    norm = np.sqrt(a**2 + b**2)
    a_norm, b_norm = a / norm, b / norm
    L = config['L']
    N = config['N']
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    from numpy.fft import fftfreq
    k = fftfreq(N, d=dx) * 2 * np.pi
    psi_base = create_bimodal_superposition(x, x1=-6.0, x2=6.0, sigma=1.0,
                                              a=a_norm, b=b_norm, phase_shift=np.pi)
    measured = []
    lower_errors = []
    upper_errors = []
    for ns in noise_strengths:
        count_left = 0
        total = num_perturbations
        for _ in range(total):
            psi_perturbed = perturb_wavefunction(psi_base, x, noise_strength=ns)
            psi_final = evolve_wavefunction(psi_perturbed, config, x, k)
            attractor, _ = identify_attractor(psi_final, x)
            if attractor == 'left':
                count_left += 1
        freq = count_left / total
        (lower, upper) = binom.interval(0.95, total, freq)
        lower_errors.append(freq - lower / total)
        upper_errors.append(upper / total - freq)
        measured.append(freq)
    noise_strengths = np.array(noise_strengths)
    measured = np.array(measured)
    lower_errors = np.array(lower_errors)
    upper_errors = np.array(upper_errors)
    plt.errorbar(noise_strengths, measured,
                 yerr=[lower_errors, upper_errors],
                 fmt='o-', color='green', capsize=3)
    plt.xlabel("Noise Strength")
    plt.ylabel("Measured Left Frequency")
    plt.title(f"Sensitivity Study for a={a_norm:.3f}, b={b_norm:.3f}")
    filename = os.path.join(results_dir, f"sensitivity_study_{meta}.png")
    plt.savefig(filename)
    plt.show()

def plot_basin_geometry(config, a, b, num_perturbations=1000, meta=""):
    """
    For a given (a, b) configuration, collects raw ⟨x⟩ values and plots a histogram (density plot)
    to visualize the basin geometry.
    """
    from simulation import create_bimodal_superposition, perturb_wavefunction, evolve_wavefunction
    norm = np.sqrt(a**2 + b**2)
    a_norm, b_norm = a / norm, b / norm
    L = config['L']
    N = config['N']
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    from numpy.fft import fftfreq
    k = fftfreq(N, d=dx) * 2 * np.pi
    psi_base = create_bimodal_superposition(x, x1=-6.0, x2=6.0, sigma=1.0,
                                              a=a_norm, b=b_norm, phase_shift=np.pi)
    x_means = []
    for _ in range(num_perturbations):
        psi_perturbed = perturb_wavefunction(psi_base, x)
        psi_final = evolve_wavefunction(psi_perturbed, config, x, k)
        rho = np.abs(psi_final)**2
        x_mean = np.trapezoid(x * rho, x)
        x_means.append(x_mean)
    plt.hist(x_means, bins=30, density=True, color='magenta', alpha=0.7)
    plt.xlabel("⟨x⟩")
    plt.ylabel("Density")
    plt.title(f"Basin Geometry for a={a_norm:.3f}, b={b_norm:.3f}")
    filename = os.path.join(results_dir, f"basin_geometry_a_{a_norm:.3f}_b_{b_norm:.3f}_{meta}.png")
    plt.savefig(filename)
    plt.show()
