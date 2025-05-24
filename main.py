import numpy as np
from simulation import multi_test_sweep_mp
from plotting import (plot_sweep_results, save_results_to_csv, export_raw_data,
                      sensitivity_study, plot_basin_geometry)

def main():
    # Configuration parameters
    config = {
        'T_param': 1.0,
        'alpha': 2.0,
        'd_tau': 0.001,
        'num_steps': 500,  # For publication, consider 2000+.
        'L': 20.0,
        'N': 1024,
        'imaginary_time': True
    }

    # Flags for optional functions
    RUN_SENSITIVITY = True
    RUN_BASIN_GEOMETRY = True

    # Metadata string for filenames
    meta = f"T{config['T_param']}_steps{config['num_steps']}"

    # Define parameter sweep ranges for pre-normalization a and b values
    a_values = np.linspace(0.2, 1.0, 10)
    b_values = np.linspace(0.2, 1.0, 10)

    # Run the simulation using the multiprocessing sweep
    results_grid, raw_data_grid = multi_test_sweep_mp(config, a_values, b_values, num_perturbations=100)

    # Plot and export the aggregated results
    plot_sweep_results(results_grid, meta=meta)
    save_results_to_csv(results_grid, meta=meta)
    export_raw_data(raw_data_grid, meta=meta)

    # Optionally run sensitivity study and basin geometry visualization
    if RUN_SENSITIVITY:
        sensitivity_noise = np.linspace(0.01, 0.1, 10)
        sensitivity_study(config, a=0.8, b=0.6, noise_strengths=sensitivity_noise,
                          num_perturbations=100, meta=meta)
    if RUN_BASIN_GEOMETRY:
        plot_basin_geometry(config, a=0.8, b=0.6, num_perturbations=1000, meta=meta)

if __name__ == "__main__":
    main()
