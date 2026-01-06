import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import SplineTransformer
from scipy.integrate import trapezoid
from dataclasses import dataclass, field, InitVar
from typing import List, Optional, Tuple
from workout import WorkoutConfig, WorkoutPhase, load_preprocess_and_filter, plot_raw_data, apply_workout_grid
from pathlib import Path
import seaborn as sns
@dataclass(frozen=True)
class SessionMetrics:
    """Immutable record of a training session's performance metrics."""
    session_id: str
    initial_punch: float
    final_punch: float
    hrr_60: float
    cardiac_costs: List[float]  # Total beats per interval
    drift_ratio: float          # Ratio of final interval cost to first



def fit_bayesian_spline(time, hr_data, total_duration, degree):
    """
    Factored model definition. 
    Returns the basis matrix (B), its derivative (dB_dt), and the MAP weights.
    """
    # 1. Create Basis
    knots = np.array(range(0, total_duration+1, 60)).reshape((-1, 1))
    transformer = SplineTransformer(knots=knots, degree=degree, include_bias=True)
    B = transformer.fit_transform(time.reshape(-1, 1))
    
    # 2. Compute Derivative Basis
    # We use gradient here to get the slope of the basis functions
    dB_dt = np.gradient(B, axis=0)

    # 3. Bayesian Model Definition
    with pm.Model() as model:
        w = pm.Normal("w", mu=130, sigma=40, shape=B.shape[1])
        mu = pm.Deterministic("mu", pm.math.dot(B, w))
        # Derivative/Punch at every time step
        accel = pm.Deterministic("accel", pm.math.dot(w, dB_dt.T))
        
        sigma = pm.HalfNormal("sigma", sigma=5)
        if hr_data is None:
            pm.Normal("obs", mu=mu, sigma=sigma)
        else:
            pm.Normal("obs", mu=mu, sigma=sigma, observed=hr_data)
        
        # Sampling 1000 draws to build the HDI
        trace = pm.sample(500, tune=1000, chains=2, target_accept=0.9, progressbar=False)
        
    return trace, B

from dataclasses import dataclass, field

@dataclass
class TreadmillAnalytic:
    time: np.ndarray
    hr: Optional[np.ndarray]
    session_id: str
    degree: int
    config: WorkoutConfig

    # Storage for processed curves
    n_intervals: int =  field(init=False, repr=False)
    trace: az.InferenceData = field(init=False, repr=False)
    post_mu: np.ndarray = field(init=False, repr=False)
    mu_mean: np.ndarray = field(init=False, repr=False)
    hdi_data: np.ndarray = field(init=False, repr=False)
    post_accel: np.ndarray = field(init=False, repr=False)
    load_cache: InitVar[bool] = False
    cache_path: InitVar[Path | None] = None

    def __post_init__(self, load_cache, cache_path):
        """Coordinates the model fitting and signal generation."""

        self.n_intervals = len(self.config.get_windows("high"))
        if load_cache and cache_path is not None and cache_path.exists():
            self.trace = az.from_netcdf(cache_path)
        else:
            self.trace, _ = fit_bayesian_spline(self.time, self.hr, self.config.total_duration, self.degree)
            self.trace.to_netcdf(cache_path)
        # Extract posterior data
        self.post_mu = az.extract(self.trace, var_names="mu").values.T
        self.mu_mean = self.post_mu.mean(axis=0)
        self.hdi_data = az.hdi(self.trace, hdi_prob=0.89).mu.values
        self.post_accel = az.extract(self.trace, var_names="accel").values.T

    def get_acceleration_peaks(self) -> List[Tuple[WorkoutPhase, float, float, float]]:
        results = []
        for phase, (phase_start, phase_end) in self.config.iter_phases():
            win = (self.time >= phase_start) & (self.time <= phase_end)
            # Get the max for EVERY sample in the window (Result: 2000 max values)
            if phase.intensity == "high":
                sample_peaks = self.post_accel[:, win].max(axis=1)
            elif phase.intensity == "low" or phase.intensity == "cooldown":
                sample_peaks = self.post_accel[:, win].min(axis=1)
            else:
                raise ValueError(f"Unknown intensity: {phase.intensity}")

            # Compute the mean and HDI of the MAXES
            punch_mean = np.median(sample_peaks, axis=0)
            punch_hdi = az.hdi(sample_peaks, hdi_prob=0.89)

            results.append((phase, punch_mean, punch_hdi[0], punch_hdi[1]))

        return results

    def get_hrr60(self) -> Tuple[WorkoutPhase, float, float, float]:
        for phase, (phase_start, phase_end) in self.config.iter_phases():
            if phase.name != "Cd":
                continue
            delta_dist = 60 * (self.post_mu[:, phase_start] - self.post_mu[:, phase_end]) / phase.duration_sec
            hrr_60_mean = np.median(delta_dist, axis=0)
            hrr_60_hdi = az.hdi(delta_dist, hdi_prob=0.89)
            return (phase, hrr_60_mean, hrr_60_hdi[0], hrr_60_hdi[1])
        raise ValueError("No cooldown phase defined in config.")

    def get_cardiac_costs(self) -> List[Tuple[WorkoutPhase, float, float, float]]:
        
        costs = []
        # Get the window for each phase
        for phase, (phase_start, phase_end) in self.config.iter_phases():
            win = (self.time >= phase_start) & (self.time <= phase_end)
            # 2. Calculate the cost for every single sample (HR is in BPM, so we divide by 60)
            # This gives you a distribution of 2000 'Total Beats' values
            cardiac_costs_dist = np.trapezoid(self.post_mu[:, win], self.time[win], axis=1) / 60
            # 3. Store the mean and HDI for the report
            cardiac_cost_mean = np.median(cardiac_costs_dist, axis=0)
            cardiac_cost_hdi = az.hdi(cardiac_costs_dist, hdi_prob=0.89)

            costs.append((phase, cardiac_cost_mean, cardiac_cost_hdi[0], cardiac_cost_hdi[1]))
        return costs

    @staticmethod
    def results_to_df(results: List[Tuple[WorkoutPhase, float, float, float]]) -> pd.DataFrame:
        return pd.DataFrame([(i[0].name, i[1], i[2], i[3]) for i in results],
                             columns=["Interval", "mean", "low", "high"])

    def get_quadrant_coords(self) -> Tuple[Tuple[WorkoutPhase, float, float, float], 
                                           Tuple[WorkoutPhase, float, float, float], 
                                           Tuple[WorkoutPhase, float, float, float]]:
        """Calculates means and 89% HDI for first Punch and HRR-60."""
        # phase, punch_mean, punch_hdi[0], punch_hdi[1]
        punches = [p for p in self.get_acceleration_peaks() if p[0].intensity == "high"]
        return punches[0], punches[-1], self.get_hrr60()


@dataclass
class TreadmillReport:
    workout_structure: WorkoutConfig

    # time_axis: np.ndarray = field(init=False)
    sessions: List[TreadmillAnalytic] = field(init=False)
    accel_results: List[Tuple[WorkoutPhase, float, float, float]] = field(init=False)
    hrr60_results: List[Tuple[WorkoutPhase, float, float, float]] = field(init=False)
    cardiac_cost_results: List[Tuple[WorkoutPhase, float, float, float]] = field(init=False)
    data_directory: InitVar[List[Path]]
    with_prior: bool

    def __post_init__(self, file_paths: List[Path]):
        degree = 3
        session_names = [p.stem[6:16] for p in file_paths]
        cache_paths = [file_path.with_suffix(".nc") for file_path in file_paths]
        time_axis, raw_data_matrix = load_preprocess_and_filter(file_paths, kernel_size=1, n_drop=240)

        self.sessions = [
            TreadmillAnalytic(time_axis, hr_data, sn, degree, self.workout_structure, True, cp) 
            for hr_data, sn, cp in zip(raw_data_matrix, session_names, cache_paths)
        ]
        if self.with_prior:
            self.sessions.append(
                TreadmillAnalytic(time_axis, None, "2026-02-01_00-00-00", degree, self.workout_structure, True, Path("prior.nc"))
            )
        assert len(file_paths) == len(session_names)
        assert len(file_paths) == raw_data_matrix.shape[0]
        np.any(np.isnan(raw_data_matrix))

    def plot_raw_data(self):
        common_time = self.sessions[0].time
        raw_data = np.array([s.hr for s in self.sessions if s.hr is not None])
        plot_raw_data(common_time, raw_data, [s.session_id for s in self.sessions], self.workout_structure)
        

    def plot_fit_accel(self, fig, axes):
        n_sessions = len(axes[1])
        for i, session in enumerate(self.sessions[-n_sessions:]):
            ax1, ax2 = axes[0, i], axes[1, i]
            posterior_mu = session.post_mu
            mu_median = np.median(posterior_mu, axis=0)
            mu_hdi = az.hdi(session.trace, hdi_prob=0.89).mu.values
            
            # Calculate Acceleration (Derivative)
            posterior_accel = session.post_accel
            accel_median = np.median(posterior_accel, axis=0)
            accel_hdi = az.hdi(posterior_accel, hdi_prob=0.89)

            if session.hr is not None:
                ax1.scatter(session.time, session.hr, color='black', s=1, alpha=0.2)
            ax1.plot(session.time, mu_median, color='firebrick', linewidth=2.5, label='Posterior Median HR')
            ax1.fill_between(session.time, mu_hdi[:, 0], mu_hdi[:, 1], color='firebrick', alpha=0.2)
            ax1.set_ylabel("Heart Rate (bpm)")
            ax1.set_title(session.session_id)
            apply_workout_grid(ax1, self.workout_structure)
            # ax1.legend(loc='upper right')

            # --- Bottom Plot: Acceleration ---
            ax2.plot(session.time, accel_median, color='indigo', linewidth=2, label='HR Acceleration')
            ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax2.fill_between(session.time, accel_hdi[:, 0], accel_hdi[:, 1], color='firebrick', alpha=0.2)
            ax2.set_ylabel("Acceleration ($\\Delta$BPM/sec)", fontweight='bold')
            ax2.set_xlabel("Time (seconds)", fontweight='bold')
            apply_workout_grid(ax2, self.workout_structure)
            # ax2.legend(loc='upper right')
    
    def plot_trend_vector(self, fig, ax, code):
        """Plots historical sessions with a vector arrow and session name annotations."""
        # 1. Collect all coordinates
        packed_coords = [s.get_quadrant_coords() for s in self.sessions]
        if code == 'ph':
            coords = [(ip[1], h[1], (ip[2], ip[3]), (h[2], h[3])) 
                      for ip, lp, h in packed_coords]
        elif code == 'pp':
            coords = [(ip[1], lp[1], (ip[2], ip[3]), (lp[2], lp[3])) 
                      for ip, lp, h in packed_coords]
        else:
            raise ValueError(f"Unknown code: {code}")
        
        for i, (p_mu, h_mu, p_hdi, h_hdi) in enumerate(coords):
            # 2. Plot the HDI Crosshairs (Same color: teal)
            ax.errorbar(p_mu, h_mu, 
                        xerr=[[p_mu - p_hdi[0]], [p_hdi[1] - p_mu]],
                        yerr=[[h_mu - h_hdi[0]], [h_hdi[1] - h_mu]], # Assuming h_mean is h_mu from context
                        fmt='o', color='teal', alpha=0.5, capsize=0, elinewidth=1.2, markersize=4)
            
            # 3. Add Session Name Annotation
            # We offset the text slightly to the right (xytext)
            ax.annotate(
                self.sessions[i].session_id, 
                xy=(p_mu, h_mu), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=5,
                fontweight='medium',
                color='#333333'
            )

            # 4. Draw the Vector Arrow
            if i > 0:
                prev_p, prev_h, _, _ = coords[i-1]
                arrow = FancyArrowPatch(
                    (prev_p, prev_h), (p_mu, h_mu),
                    arrowstyle='-|>', 
                    mutation_scale=15, 
                    color='teal', # Matching point color
                    linestyle='-',
                    linewidth=1.5, 
                    alpha=0.3,
                    shrinkA=5, # Don't start exactly on the point
                    shrinkB=5  # Don't end exactly on the point
                )
                ax.add_patch(arrow)

        # Formatting
        fig.suptitle("Longitudinal Fitness Evolution")
        ax.set_xlabel("1st Punch (Max Accel - BPM/s)")
        if code == 'ph': 
            ax.set_ylabel("Recovery (HRR60 - BPM)")
        else:
            ax.set_ylabel("last Punch (Max Accel - BPM/s)")
        ax.grid(True, linestyle=':', alpha=0.4)
        

    def plot_cardiac_cost(self):
        cc_df = pd.concat([TreadmillAnalytic.results_to_df(s.get_cardiac_costs())
                            .assign(session=s.session_id)
                            .assign(session_date=lambda x: pd.to_datetime(x.session.str[:10]))
                            for s in self.sessions]).reset_index().rename(columns={'mean': 'cc'})
        ax = sns.pointplot(x='session_date', y='cc', hue='Interval', data=cc_df, legend='brief')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))