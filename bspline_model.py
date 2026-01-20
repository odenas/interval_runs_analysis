import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import SplineTransformer
from dataclasses import dataclass, field, InitVar
from typing import List, Optional, Tuple
from pathlib import Path
import seaborn as sns
from dataclasses import dataclass, field

from workout import WorkoutConfig, WorkoutPhase


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


@dataclass
class KnotGenerator:
    time: np.ndarray
    degree: int

    @property
    def total_duration(self):
        return self.time.shape[0]

    def _basis(self, knots):
        transformer = SplineTransformer(knots=knots, degree=self.degree, include_bias=True)
        return transformer.fit_transform(self.time.reshape(-1, 1))

    def equally_spaced_cadence(self, knot_spacing):
        knots = np.array(range(0, self.total_duration+1, knot_spacing)).reshape((-1, 1))
        return self._basis(knots)

    def equally_spaced_n(self, n_knots):
        freq = int(self.total_duration / n_knots)
        return self.equally_spaced_cadence(freq)
    
    def custom(self, knots):
        return self._basis(knots)



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
    config: WorkoutConfig
    trace: az.InferenceData

    load_cache: InitVar[bool] = False
    cache_path: InitVar[Path | None] = None

    # Storage for processed curves
    post_mu: np.ndarray = field(init=False, repr=False)  # samples x seconds
    post_accel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, load_cache, cache_path):
        """Coordinates the model fitting and signal generation."""

        n_samples = 1000
        # if load_cache and cache_path is not None and cache_path.exists():
        #     self.trace = az.from_netcdf(cache_path)
        # else:
        #     self.trace, _ = fit_bayesian_spline(self.hr, self.knot_basis, n_samples)
        #     self.trace.to_netcdf(cache_path)

        # Extract posteriors
        self.post_mu = az.extract(self.trace, var_names="mu").values.T
        self.post_accel = az.extract(self.trace, var_names="accel").values.T
        
        assert self.post_mu.shape == (n_samples, self.time.shape[0])
        assert self.post_accel.shape == (n_samples, self.time.shape[0])

    def get_acceleration_peaks(self) -> List[Tuple[WorkoutPhase, MedianHdi]]:
        """
        maximum rate of change in your heart rate when you shift phase intensity
        """
        results = []
        windows = [(start, end) for (phase, (start, end)) in self.config.iter_phases()]
        for i, phase in enumerate(self.config.phases):
            if i == 0:
                prev_center = 0
                next_center = (windows[i+1][1] + windows[i+1][0]) // 2
            elif i == len(windows) - 1:
                prev_center = (windows[i-1][1] + windows[i-1][0]) // 2
                next_center = windows[i][1]
            else:
                prev_center = (windows[i-1][1] + windows[i-1][0]) // 2
                next_center = (windows[i+1][1] + windows[i+1][0]) // 2
            
            if phase.intensity in ('cooldown', 'low'):
                peak_dist = self.post_accel[:, prev_center:next_center].min(axis=1)
            else:
                peak_dist = self.post_accel[:, prev_center:next_center].max(axis=1)
            results.append((phase, MedianHdi.from_samples(peak_dist)))
            # print(prev_center, next_center, MedianHdi.from_samples(peak_dist))

        return results

    def get_hrr60(self) -> Tuple[WorkoutPhase, MedianHdi]:
        """
        the decrease in heart rate precisely 60 seconds after the final high-intensity phase ends and the cooldown begins.
        """
        # find the last peak
        ss, se = self.config.get_windows('high')[-1]
        s_center = (ss + se) // 2
        cds, cde = self.config.get_windows('cooldown')[-1]
        cd_center = (cds + cde) // 2
        assert se == cds

        peak_arg_dist = np.argmax(self.post_mu[:, s_center:cd_center], axis=1) + s_center
        valley_arg_dist = peak_arg_dist + 60
        row_args = np.arange(self.post_mu.shape[0])
        delta_dist = self.post_mu[row_args, peak_arg_dist] - self.post_mu[row_args, valley_arg_dist]
        return (self.config.phases[-1], MedianHdi.from_samples(delta_dist))

    def get_cardiac_costs(self) -> List[Tuple[WorkoutPhase, MedianHdi]]:
        """
        the total number of heartbeats spent during a specific phase"""
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


@dataclass
class TreadmillReport:
    workout_structure: WorkoutConfig
    sessions: List[TreadmillAnalytic]

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
        try:
            n_sessions = len(axes[1])
        except TypeError:
                n_sessions = 1
        for i, session in enumerate(self.sessions[-n_sessions:]):
            if n_sessions == 1:
                ax1, ax2 = axes[0], axes[1]
            else:
                ax1, ax2 = axes[0, i], axes[1, i]
            hr_stats = MedianHdiSample.from_samples(session.post_mu, session.post_mu.shape[0])
            acc_stats = MedianHdiSample.from_samples(session.post_accel, session.post_accel.shape[0])

            # --- Top Plot: Heart Rate ---
            if session.hr is not None:
                ax1.scatter(session.time, session.hr, color='black', s=1, alpha=0.2)
            ax1.plot(session.time, hr_stats.median, color="#2b88b4", linewidth=1, label='Posterior Median HR')
            ax1.fill_between(session.time, hr_stats.hdi_lower, hr_stats.hdi_upper, color="#2b88b4", alpha=0.2)
            ax1.set_ylabel("Heart Rate (bpm)")
            ax1.set_title(session.session_id)
            apply_workout_grid(ax1, self.workout_structure)


            # --- Bottom Plot: Acceleration ---
            ax2.plot(session.time, acc_stats.median, color="#2b88b4", linewidth=1, label='HR Acceleration')
            ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax2.fill_between(session.time, acc_stats.hdi_lower, acc_stats.hdi_upper, color="#2b88b4", alpha=0.2)
            ax2.set_ylabel("Acceleration ($\\Delta$BPM/sec)", fontweight='bold')
            ax2.set_xlabel("Time (seconds)", fontweight='bold')
            # add peak markers            
            for phase_iter, phase_peaks in zip(session.config.iter_phases(), session.get_acceleration_peaks()):
                phase: WorkoutPhase = phase_peaks[0]
                peak: MedianHdi = phase_peaks[1]
                start, end = phase_iter[1]
                if phase.intensity in ('low', 'high'):
                    ax2.plot([start, end], [peak.hdi_lower, peak.hdi_lower], color="#0b0c0c", linewidth=0.2, alpha=0.8, linestyle='-')
                    ax2.plot([start, end], [peak.hdi_upper, peak.hdi_upper], color="#0b0c0c", linewidth=0.2, alpha=0.8, linestyle='-')
                    # ax2.errorbar(x=(start+ end) / 2, y=peak.median, yerr=peak.hdi_width,
                    #              fmt='o', color="#0b0c0c", capsize=0, elinewidth=.5, markersize=2)
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
            if i == 0:
                ax.errorbar(x_mu, y_mu, xerr=[x_err[1]], yerr=[y_err[1]],
                            fmt='o', color="#7c4e0c", capsize=0, elinewidth=.5, markersize=4)

            elif i > 0 and i < len(self.sessions) - 1:
                ax.errorbar(x_mu, y_mu, xerr=[x_err[0]], yerr=[y_err[0]],
                            fmt='s', color="#645b4e", capsize=0, elinewidth=.5, markersize=2)
            if i == len(self.sessions) - 1:
                ax.errorbar(x_mu, y_mu, xerr=[x_err[1]], yerr=[y_err[1]],
                            fmt='o', color="#7c4e0c", capsize=0, elinewidth=.5, markersize=4)

            ax.annotate(session.session_id, xy=(x_mu, y_mu), 
                        xytext=(5, 5), textcoords='offset points', fontsize=5, color='#333333')
            
            if i > 0:
                prev_x, prev_y, _, _ = all_coords[i-1]
                arrow = FancyArrowPatch((prev_x, prev_y), (x_mu, y_mu),
                                        arrowstyle='-|>', mutation_scale=15, color='grey',
                                        linestyle='-', linewidth=.5, alpha=0.4, shrinkA=5, shrinkB=5)
                ax.add_patch(arrow)

        # Formatting
        fig.suptitle("Longitudinal Fitness Evolution")
        ax.set_xlabel("1st Punch (Max Accel - BPM/s)")
        if code == 'ph': 
            ax.set_ylabel("Recovery (HRR60 - BPM)")
        else:
            ax.set_ylabel("Last Punch (Max Accel - BPM/s)")
        ax.grid(True, linestyle=':', alpha=0.2)
        

    def plot_cardiac_cost(self):
        cc_df = pd.concat([TreadmillAnalytic.results_to_df(s.get_cardiac_costs())
                            .assign(session=s.session_id)
                            .assign(session_date=lambda x: pd.to_datetime(x.session.str[:10]))
                            for s in self.sessions]).reset_index().rename(columns={'mean': 'cc'})
        g = sns.catplot(x='session_date', y='cc', hue='Interval', data=cc_df, legend='brief', height=4, aspect=2.5, kind='point')
        # ax.set_ylabel("Cardiac Cost (Total Beats)")
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))