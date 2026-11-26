"""
RunPod GPU Availability Simulator

Models GPU availability on RunPod Secure Cloud WITHOUT knowing actual stock count.

Based on:
- 77 pricing-API distribution summaries (11 March to 20 April 2026, Taiwan time)
- Cloud GPU availability patterns
- Time-of-day demand variation
- Queue wait time modeling

Key insight: We don't know RunPod's actual GPU inventory, but we can model
availability patterns and queue times based on cloud behavior.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

class StockStatus(str, Enum):
    """Qualitative GPU stock status."""
    HIGH = "High"     # Plenty of GPUs available, near-zero wait
    MEDIUM = "Medium" # Moderate availability, short wait
    LOW = "Low"       # Limited availability, significant queue likely

class TimeOfDay(Enum):
    """Time periods with different availability patterns."""
    OFF_PEAK = "off_peak"    # 12am - 6am (low demand)
    MORNING = "morning"        # 6am - 12pm (moderate demand)
    PEAK = "peak"             # 12pm - 6pm (high demand)
    EVENING = "evening"        # 6pm - 12am (moderate demand)


@dataclass
class AvailabilityState:
    """
    Current availability state for a GPU type.
    """
    gpu_type: str
    available_now: bool
    estimated_queue_time: float  # minutes
    probability_available: float


class RunPodAvailabilityModel:
    """
    Models RunPod GPU availability based on qualitative StockStatus.

    Provides "High", "Medium", or "Low" stock status based on time-of-day demand,
    mirroring the format of real RunPod pricing data.
    """

    # Probability of HIGH stock for each GPU type
    BASE_HIGH_STOCK_PROB = {
        "RTX_3090": 0.50,
        "RTX_A5000": 0.65,
        "RTX_4090": 0.30,
        "RTX_A4500": 0.70,
        "RTX_A4000": 0.75,
    }

    # Time-of-day multipliers for HIGH/MEDIUM stock probability
    STOCK_PROB_MULTIPLIERS = {
        TimeOfDay.OFF_PEAK: 1.3,
        TimeOfDay.MORNING: 0.9,
        TimeOfDay.PEAK: 0.5,
        TimeOfDay.EVENING: 1.0,
    }

    # Wait time ranges (seconds) based on StockStatus
    WAIT_TIME_RANGES = {
        StockStatus.HIGH: (0.0, 10.0),       # High stock: 0-10s wait
        StockStatus.MEDIUM: (30.0, 120.0),   # Medium stock: 30s-2m wait
        StockStatus.LOW: (600.0, 7200.0),    # Low stock: 10m-2h wait (includes empty cases)
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize availability model.
        """
        if seed is not None:
            self.random = random.Random(seed)
        else:
            self.random = random.Random(42)

    def get_time_of_day(self, timestamp: float) -> TimeOfDay:
        """Get time period from a simulation-relative timestamp.

        Simulation time zero is anchored to 08:00 on the modelled day, the
        calibrated operating point recorded in DECISIONS.md section 8. The
        offset is applied arithmetically rather than via
        ``datetime.fromtimestamp``, which would read the host's local clock and
        so make the band (and therefore every downstream result) depend on the
        timezone the run happens to execute under. The 08:00 origin, and hence
        all canonical numbers, are preserved exactly.
        """
        hour = int((timestamp + 8 * 3600) / 3600) % 24

        if 0 <= hour < 6:
            return TimeOfDay.OFF_PEAK
        elif 6 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 18:
            return TimeOfDay.PEAK
        else:
            return TimeOfDay.EVENING

    def get_stock_status(self, gpu_type: str, timestamp: float) -> StockStatus:
        """
        Get the qualitative stock status for a GPU type.

        Returns:
            StockStatus (HIGH, MEDIUM, or LOW)
        """
        base_prob = self.BASE_HIGH_STOCK_PROB.get(gpu_type, 0.5)
        time_period = self.get_time_of_day(timestamp)
        multiplier = self.STOCK_PROB_MULTIPLIERS[time_period]
        
        prob_high = min(0.95, base_prob * multiplier)
        roll = self.random.random()

        if roll < prob_high:
            return StockStatus.HIGH

        # prob_medium deliberately overflows (it is not the residual mass), so
        # prob_high + prob_medium can exceed 1 and then makes LOW unreachable.
        # Whether that happens depends entirely on which beta_g this call was
        # indexed with. For the four fleet types the simulator actually draws
        # against (RTX 3090/A5000/A4500/A4000, beta_g 0.50 to 0.75) the sum is
        # at least 1 in every band except the midday peak, so LOW is reachable
        # only during the peak, at P(LOW) 0.0625 (A4000) to 0.3750 (3090).
        # It is NOT a property of the table as a whole: RTX_4090 (beta_g 0.30)
        # is reachable in every band, P(LOW) 0.325 in the morning and 0.250 in
        # the evening. No fleet slot resolves to that entry, and
        # cloud_fleet.update_stock_statuses must keep indexing on the raw
        # resource type so that it cannot begin to.
        prob_medium = min(0.90, base_prob * multiplier * 1.5)
        if roll < (prob_high + prob_medium):
            return StockStatus.MEDIUM

        return StockStatus.LOW

    def get_provisioning_delay(self, gpu_type: str, timestamp: float, node_id: Optional[int] = None) -> float:
        """
        Calculate provisioning delay based on current stock status.
        """
        status = self.get_stock_status(gpu_type, timestamp)
        wait_min, wait_max = self.WAIT_TIME_RANGES[status]
        
        # Deterministic seed independent of PYTHONHASHSEED: Python's built-in
        # hash() of a string is randomised per process, which would make
        # provisioning delays (and hence all downstream results) irreproducible
        # across runs. zlib.crc32 of a stable string key is process-independent.
        import zlib
        key = f"{timestamp}|{node_id}|{gpu_type}".encode()
        seed_val = zlib.crc32(key)
        local_random = random.Random(seed_val)
        
        return local_random.uniform(wait_min, wait_max)

    def wait_for_availability(
        self,
        gpu_type: str,
        start_timestamp: float,
        max_wait_minutes: float = 120.0
    ) -> Tuple[float, float]:
        """
        Simulate waiting for GPU availability based on status.
        """
        delay_sec = self.get_provisioning_delay(gpu_type, start_timestamp)
        delay_sec = min(delay_sec, max_wait_minutes * 60.0)
        return start_timestamp + delay_sec, delay_sec / 60.0


class RunPodQueueSimulator:
    """
    Simulates queueing behavior for multiple GPU requests.
    """

    def __init__(self, availability_model: RunPodAvailabilityModel):
        self.availability_model = availability_model
        self.queue_log: List[Dict] = []

    def request_gpu(
        self,
        gpu_type: str,
        request_time: float,
        job_id: str
    ) -> Dict:
        available_time, wait_minutes = self.availability_model.wait_for_availability(
            gpu_type, request_time
        )

        log_entry = {
            "job_id": job_id,
            "gpu_type": gpu_type,
            "request_time": request_time,
            "available_time": available_time,
            "wait_minutes": wait_minutes,
            "wait_seconds": wait_minutes * 60
        }
        self.queue_log.append(log_entry)
        return log_entry

    def get_queue_statistics(self) -> Dict:
        if not self.queue_log:
            return {
                "total_requests": 0,
                "avg_wait_minutes": 0,
                "max_wait_minutes": 0,
                "jobs_waited": 0,
                "jobs_waited_percentage": 0
            }

        wait_times = [log["wait_minutes"] for log in self.queue_log]
        jobs_waited = sum(1 for wt in wait_times if wt > 0)

        return {
            "total_requests": len(self.queue_log),
            "avg_wait_minutes": sum(wait_times) / len(wait_times),
            "max_wait_minutes": max(wait_times),
            "min_wait_minutes": min(wait_times),
            "jobs_waited": jobs_waited,
            "jobs_waited_percentage": (jobs_waited / len(wait_times)) * 100,
            "total_wait_minutes": sum(wait_times)
        }
