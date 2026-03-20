
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class WorkoutPhase:
    name: str
    duration_sec: int
    intensity: str  # 'low', 'high', or 'cooldown'

@dataclass(frozen=True)
class WorkoutConfig:
    phases: List[WorkoutPhase]
    name: str
    
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
        for phase, (start, end) in self.iter_phases():
            if phase.intensity == intensity:
                windows.append((start, end))
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






# Your current specific workout: 3 reps of (3m low, 1m high) + 2m cooldown
standard_3x4 = WorkoutConfig(phases=[
    WorkoutPhase(f"R1", 180, 'low'), WorkoutPhase(f"S1", 60, 'high'),
    WorkoutPhase(f"R2", 180, 'low'), WorkoutPhase(f"S2", 60, 'high'),
    WorkoutPhase(f"R3", 180, 'low'), WorkoutPhase(f"S3", 60, 'high'),
    WorkoutPhase("Cd", 120, 'cooldown')
], name="standard_3x4")

l2h2_4x4 = WorkoutConfig(phases=[
    WorkoutPhase(f"R1", 120, 'low'), WorkoutPhase(f"S1", 120, 'high'),
    WorkoutPhase(f"R2", 120, 'low'), WorkoutPhase(f"S2", 120, 'high'),
    WorkoutPhase(f"R3", 120, 'low'), WorkoutPhase(f"S3", 120, 'high'),
    WorkoutPhase(f"R4", 120, 'low'), WorkoutPhase(f"S4", 120, 'high'),
    WorkoutPhase("Cd", 120, 'cooldown')
], name="l2h2_4x4")

xtrainer_4x4 = WorkoutConfig(phases=[
    WorkoutPhase(f"R1", 180, 'low'), WorkoutPhase(f"S1", 180, 'high'),
    WorkoutPhase(f"R2", 180, 'low'), WorkoutPhase(f"S2", 180, 'high'),
    WorkoutPhase(f"R3", 180, 'low'), WorkoutPhase(f"S3", 180, 'high'),
    WorkoutPhase(f"R3", 180, 'low'), WorkoutPhase(f"S3", 180, 'high'),
    WorkoutPhase("Cd", 120, 'cooldown')
], name="xtrainer_4x4")
