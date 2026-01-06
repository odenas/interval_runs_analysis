from dataclasses import dataclass, field, InitVar
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

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


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np

def plot_raw_data(common_time, raw_data, filenames, config: WorkoutConfig):
    """
    Plot HR data annotated with workout phases.
    """

    n_sessions = raw_data.shape[0]
    colors = cm.get_cmap('copper', n_sessions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i in range(n_sessions):
        ax.plot(common_time, raw_data[i], color=colors(n_sessions - i - 1), alpha=0.6, linewidth=1, label=filenames[i])
    
    ax.set_title("Heart Rate Data (%d Sessions)" % n_sessions)
    ax.set_ylabel("HR (bpm)")
    ax.set_xlabel("Time (seconds)")
    apply_workout_grid(ax, config)    
    # ax.legend()
    return fig, ax
