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
from pathlib import Path
import seaborn as sns
from dataclasses import dataclass, field
import matplotlib.cm as cm
from scipy.signal import medfilt

@dataclass(frozen=True)
class WorkoutPhase:
    name: str
    duration_sec: int
    intensity: str  # 'low', 'high', or 'cooldown'

@dataclass(frozen=True)
class WorkoutConfig:
    phases: List[WorkoutPhase]
    
    @property
    def total_duration(self) -> int:
        return sum(p.duration_sec for p in self.phases)


    def iter_phases(self):
        current_time = 0
        for phase in self.phases:
            yield phase, (current_time, current_time + phase.duration_sec)
            current_time += phase.duration_sec


    def get_windows(self, intensity: str) -> List[Tuple[int, int]]:
        """Calculates (start, end) timestamps for all intervals of given intensity."""
        windows = []
        current_time = 0
        for phase in self.phases:
            if phase.intensity == intensity:
                windows.append((current_time, current_time + phase.duration_sec))
            current_time += phase.duration_sec
        return windows

    def get_cooldown_window(self) -> Tuple[int, int]:
        """Calculates the recovery window (start of cooldown to 60s in)."""
        current_time = 0
        for phase in self.phases:
            # We look for the start of the final 'cooldown' phase
            if phase.intensity == 'cooldown':
                return (current_time, current_time + 60)
            current_time += phase.duration_sec
        raise ValueError("No cooldown phase defined in config.")
    



def read_csv(file):        
    df = pd.read_csv(file, skiprows=2).iloc[:, 1:3]
    if '2025-12-24' in file.stem:
        pad_df = df.head(110)
        df = pd.concat([pad_df, df])
    td = pd.to_timedelta(df['Time'])
    df['Time_sec'] = td.dt.total_seconds()
    return df



def load_preprocess_and_filter(file_paths, kernel_size=11, n_drop=0):
    """
    1. Loads CSVs and drops the first N data points.
    2. Resets time so the first remaining point is t=0.
    3. Interpolates to a common grid and applies a Median Filter.
    """
    processed_dfs = []
    max_duration = 0
    
    # Pass 1: Load, Drop, and find global max time
    for file in file_paths:
        df = read_csv(file)
        
        # --- Constraint: Drop first N points ---
        if n_drop > 0:
            df = df.iloc[n_drop:].reset_index(drop=True)
            
        # Convert Time to seconds
        df['Time_sec'] = pd.to_timedelta(df['Time']).dt.total_seconds()
        
        # Shift Time_sec so that the first point after dropping is t=0
        # This ensures all sessions align at the same relative start point
        df['Time_sec'] = df['Time_sec'] - df['Time_sec'].iloc[0]
        
        if df['Time_sec'].max() > max_duration:
            max_duration = df['Time_sec'].max()
            
        processed_dfs.append(df)
    
    common_time = np.arange(0, int(max_duration) + 1, 1)
    data_list = []
    
    for df in processed_dfs:
        # Interpolate HR onto the common grid
        hr_interp = np.interp(common_time, df['Time_sec'], df['HR (bpm)'])
        
        # Apply Median Filtering to handle Type 1 errors (substantial jitters)
        hr_cleaned = medfilt(hr_interp, kernel_size=kernel_size)
        
        data_list.append(hr_cleaned)
        
    return common_time, np.array(data_list)



@dataclass(frozen=True)
class MedianHdi:
    median: float
    hdi: Tuple[float, float]


    @classmethod
    def from_samples(cls, samples, hdi_prob=0.89):
        return cls(np.median(samples, axis=0), tuple(az.hdi(samples, hdi_prob=hdi_prob)))

    @property
    def hdi_width(self):
        return self.hdi[1] - self.hdi[0]
    @property
    def hdi_lower(self):
        return self.hdi[0]
    @property
    def hdi_upper(self):
        return self.hdi[1]

@dataclass(frozen=True)
class MedianHdiSample:
    median: np.ndarray
    hdi: np.ndarray

    @classmethod
    def from_samples(cls, samples, n_samples, hdi_prob=0.89):
        assert samples.shape[0] == n_samples
        return cls(np.median(samples, axis=0), az.hdi(samples, hdi_prob=hdi_prob))

    @property
    def hdi_width(self):
        return self.hdi[:, 1] - self.hdi[:, 0]
    @property
    def hdi_lower(self):
        return self.hdi[:, 0]
    @property
    def hdi_upper(self):
        return self.hdi[:, 1]


def fit_bayesian_spline(time, hr_data, total_duration, degree, sample_size, knot_every):
    """
    Factored model definition. 
    Returns the basis matrix (B), its derivative (dB_dt), and the MAP weights.
    """
    # 1. Create Basis
    knots = np.array(range(0, total_duration+1, knot_every)).reshape((-1, 1))
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
        chains = 2
        trace = pm.sample(int(sample_size / chains), tune=1000, chains=chains, target_accept=0.9, progressbar=False)
        
    return trace, B





def apply_workout_grid(ax, config: WorkoutConfig):
    """
    Overlays the workout structure onto a heart rate plot.
    Shades 'high' intensity regions and labels each phase.
    """

    y_min, y_max = ax.get_ylim()

    xticks = []    
    for phase, (start, end) in config.iter_phases():
        if phase.intensity == 'low' or phase.intensity == 'cooldown': 
            xticks.append(start)
        # 1. Shade the High Intensity windows
        if phase.intensity == 'high':
            ax.axvspan(start, end, color='red', alpha=0.05, label='_nolegend_')
            
        # 2. Draw vertical transition lines
        ax.axvline(start, color='grey', linestyle='--', alpha=0.3, linewidth=1)
        
        # 3. Add Phase Labels at the top of the plot
        midpoint = start + (phase.duration_sec / 2)
        # Only label if the phase is long enough to avoid clutter
        if phase.duration_sec >= 60:
            ax.text(midpoint, y_max * 0.98, phase.name, 
                    fontsize=8, ha='center', color='grey', alpha=0.7)
    # Mark the final boundary
    ax.axvline(config.total_duration, color='grey', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xticks(xticks + [config.total_duration])



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
    post_mu: np.ndarray = field(init=False, repr=False)  # samples x seconds
    post_accel: np.ndarray = field(init=False, repr=False)
    load_cache: InitVar[bool] = False
    cache_path: InitVar[Path | None] = None

    def __post_init__(self, load_cache, cache_path):
        """Coordinates the model fitting and signal generation."""

        self.n_intervals = len(self.config.get_windows("high"))
        n_samples = 1000
        if load_cache and cache_path is not None and cache_path.exists():
            self.trace = az.from_netcdf(cache_path)
        else:
            self.trace, _ = fit_bayesian_spline(self.time, self.hr, self.config.total_duration, self.degree, n_samples, 30)
            self.trace.to_netcdf(cache_path)
        
        # Extract posteriors
        self.post_mu = az.extract(self.trace, var_names="mu").values.T
        self.post_accel = az.extract(self.trace, var_names="accel").values.T
        
        assert self.post_mu.shape == (n_samples, self.time.shape[0])
        assert self.post_accel.shape == (n_samples, self.time.shape[0])

    def get_acceleration_peaks(self) -> List[Tuple[WorkoutPhase, MedianHdi]]:
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
            results.append((phase, MedianHdi.from_samples(sample_peaks)))
        return results

    def get_hrr60(self) -> Tuple[WorkoutPhase, MedianHdi]:
        for phase, (phase_start, phase_end) in self.config.iter_phases():
            if phase.name != "Cd":
                continue
            delta_dist = 60 * (self.post_mu[:, phase_start] - self.post_mu[:, phase_end]) / phase.duration_sec
            return (phase, MedianHdi.from_samples(delta_dist))
        raise ValueError("No cooldown phase defined in config.")

    def get_cardiac_costs(self) -> List[Tuple[WorkoutPhase, MedianHdi]]:
        costs = []
        # Get the window for each phase
        for phase, (phase_start, phase_end) in self.config.iter_phases():
            win = (self.time >= phase_start) & (self.time <= phase_end)
            # 2. Calculate the cost for every single sample (HR is in BPM, so we divide by 60)
            # This gives you a distribution of 2000 'Total Beats' values
            cardiac_costs_dist = np.trapezoid(self.post_mu[:, win], self.time[win], axis=1) / 60
            costs.append((phase, MedianHdi.from_samples(cardiac_costs_dist)))
        return costs

    @staticmethod
    def results_to_df(results: List[Tuple[WorkoutPhase, MedianHdi]]) -> pd.DataFrame:
        return pd.DataFrame([(phase.name, metric.median, metric.hdi_lower, metric.hdi_upper) for phase, metric in results],
                             columns=["Interval", "mean", "low", "high"])

    def get_quadrant_coords(self) -> Tuple[Tuple[WorkoutPhase, MedianHdi], 
                                           Tuple[WorkoutPhase, MedianHdi], 
                                           Tuple[WorkoutPhase, MedianHdi]]:
        # phase, punch_mean, punch_hdi[0], punch_hdi[1]
        punches = [p for p in self.get_acceleration_peaks() if p[0].intensity == "high"]
        return punches[0], punches[-1], self.get_hrr60()


@dataclass
class TreadmillReport:
    workout_structure: WorkoutConfig

    # time_axis: np.ndarray = field(init=False)
    sessions: List[TreadmillAnalytic] = field(init=False)
    accel_results: List[Tuple[WorkoutPhase, MedianHdi]] = field(init=False)
    hrr60_results: List[Tuple[WorkoutPhase, MedianHdi]] = field(init=False)
    cardiac_cost_results: List[Tuple[WorkoutPhase, MedianHdi]] = field(init=False)
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
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for session in self.sessions:
            if session.hr is None:
                continue
            ax.plot(session.time, session.hr, color='gray', linewidth=1, label=session.session_id)

        ax.set_title("Heart Rate Data (%d Sessions)" % len(self.sessions))
        ax.set_ylabel("HR (bpm)")
        ax.set_xlabel("Time (seconds)")
        apply_workout_grid(ax, self.workout_structure)    
        

    def plot_fit_accel(self, fig, axes):
        n_sessions = len(axes[1])
        for i, session in enumerate(self.sessions[-n_sessions:]):
            ax1, ax2 = axes[0, i], axes[1, i]
            hr_stats = MedianHdiSample.from_samples(session.post_mu, session.post_mu.shape[0])
            acc_stats = MedianHdiSample.from_samples(session.post_accel, session.post_accel.shape[0])

            # --- Top Plot: Heart Rate ---
            if session.hr is not None:
                ax1.scatter(session.time, session.hr, color='black', s=1, alpha=0.2)
            ax1.plot(session.time, hr_stats.median, color='firebrick', linewidth=2.5, label='Posterior Median HR')
            ax1.fill_between(session.time, hr_stats.hdi_lower, hr_stats.hdi_upper, color='firebrick', alpha=0.2)
            ax1.set_ylabel("Heart Rate (bpm)")
            ax1.set_title(session.session_id)
            apply_workout_grid(ax1, self.workout_structure)

            # --- Bottom Plot: Acceleration ---
            ax2.plot(session.time, acc_stats.median, color='indigo', linewidth=2, label='HR Acceleration')
            ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax2.fill_between(session.time, acc_stats.hdi_lower, acc_stats.hdi_upper, color='firebrick', alpha=0.2)
            ax2.set_ylabel("Acceleration ($\\Delta$BPM/sec)", fontweight='bold')
            ax2.set_xlabel("Time (seconds)", fontweight='bold')
            apply_workout_grid(ax2, self.workout_structure)

    
    def plot_trend_vector(self, fig, ax, code):
        """Plots historical sessions with a vector arrow and session name annotations."""

        all_coords = []
        for s in self.sessions:
            punches = [punch for (phase, punch) in s.get_acceleration_peaks() if phase.intensity == 'high']
            hrr60 = s.get_hrr60()[1]
            x_stat = punches[0]
            y_stat = (hrr60 if code == 'ph' else punches[-1])
            all_coords.append((x_stat.median, y_stat.median, 
                               (x_stat.median - x_stat.hdi_lower, x_stat.hdi_upper - x_stat.median), 
                               (y_stat.median - y_stat.hdi_lower, y_stat.hdi_upper - y_stat.median)))

        for i, (session, (x_mu, y_mu, x_err, y_err)) in enumerate(zip(self.sessions, all_coords)):
            ax.errorbar(x_mu, y_mu, xerr=[x_err[0]], yerr=[y_err[0]],
                        fmt='o', color='teal', alpha=0.5, capsize=0, elinewidth=1.2, markersize=4)
            ax.annotate(session.session_id, xy=(x_mu, y_mu), 
                        xytext=(5, 5), textcoords='offset points', fontsize=5, color='#333333')
            
            if i > 0:
                prev_x, prev_y, _, _ = all_coords[i-1]
                arrow = FancyArrowPatch((prev_x, prev_y), (x_mu, y_mu),
                                        arrowstyle='-|>', mutation_scale=15, color='teal',
                                        linestyle='-', linewidth=1.5, alpha=0.3, shrinkA=5, shrinkB=5)
                ax.add_patch(arrow)

        # Formatting
        fig.suptitle("Longitudinal Fitness Evolution")
        ax.set_xlabel("1st Punch (Max Accel - BPM/s)")
        if code == 'ph': 
            ax.set_ylabel("Recovery (HRR60 - BPM)")
        else:
            ax.set_ylabel("Last Punch (Max Accel - BPM/s)")
        ax.grid(True, linestyle=':', alpha=0.4)
        

    def plot_cardiac_cost(self):
        cc_df = pd.concat([TreadmillAnalytic.results_to_df(s.get_cardiac_costs())
                            .assign(session=s.session_id)
                            .assign(session_date=lambda x: pd.to_datetime(x.session.str[:10]))
                            for s in self.sessions]).reset_index().rename(columns={'mean': 'cc'})
        ax = sns.pointplot(x='session_date', y='cc', hue='Interval', data=cc_df, legend='brief')
        ax.set_ylabel("Cardiac Cost (Total Beats)")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))