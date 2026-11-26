"""
CGRSP Availability Models (unused: not imported by main.py, any phase*.py, or
run_all.sh; the live pipeline uses RunPodAvailabilityModel from
runpod_availability.py instead, the stock-status/phi(t)/beta_g model
described in the thesis). Kept for reference; do not treat as describing the
thesis's actual availability model.

Models stochastic GPU availability with time-varying customer demand rates.

Simplified model focusing on customer usage patterns only.
Provider-handled maintenance and failures are not modeled.

Based on the formulation:
  - λ_customer(t): Time-varying customer demand rate
  - μ_release: Resource release rate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List
from enum import Enum


class AvailabilityState(Enum):
    """GPU availability states."""
    AVAILABLE = "available"  # Idle and ready for jobs
    UNAVAILABLE = "unavailable"  # Rented by another customer


@dataclass
class AvailabilityConfig:
    """
    Configuration for customer-only stochastic availability model.

    Transition rates:
      - λ_customer: Customer demand arrival rate (makes GPU unavailable)
      - μ_release: Resource release rate (makes GPU available when customer finishes)

    Time-varying parameters:
      - Peak hours: Higher demand during business hours
    """
    lambda_customer: float = 0.01  # Customer demand rate (per second)
    mu_release: float = 0.02  # Release rate (per second)

    # Time-varying parameters
    peak_hours_start: int = 9  # 9AM
    peak_hours_end: int = 17  # 5PM
    peak_demand_multiplier: float = 2.0  # 2x demand during peak hours
    off_peak_multiplier: float = 0.5  # 0.5x demand off-peak

    def __post_init__(self):
        """Validate configuration."""
        if self.lambda_customer < 0:
            raise ValueError(
                f"lambda_customer must be >= 0, got {self.lambda_customer}"
            )

        if self.mu_release < 0:
            raise ValueError(f"mu_release must be >= 0, got {self.mu_release}")

        if not 0 <= self.peak_hours_start <= 23:
            raise ValueError(
                f"peak_hours_start must be in [0, 23], got {self.peak_hours_start}"
            )

        if not 0 <= self.peak_hours_end <= 23:
            raise ValueError(
                f"peak_hours_end must be in [0, 23], got {self.peak_hours_end}"
            )

        if self.peak_demand_multiplier <= 0:
            raise ValueError(
                f"peak_demand_multiplier must be > 0, "
                f"got {self.peak_demand_multiplier}"
            )

        if self.off_peak_multiplier <= 0:
            raise ValueError(
                f"off_peak_multiplier must be > 0, "
                f"got {self.off_peak_multiplier}"
            )


# Default availability configuration
DEFAULT_AVAILABILITY_CONFIG = AvailabilityConfig()


class AvailabilityModel:
    """
    Models stochastic GPU availability as a continuous-time Markov chain.

    Simplified customer-only model:
    States: AVAILABLE ↔ UNAVAILABLE
    Transitions:
      - AVAILABLE → UNAVAILABLE: λ_customer(t) (another customer rents the GPU)
      - UNAVAILABLE → AVAILABLE: μ_release (customer finishes renting)

    This model focuses on customer usage patterns only.
    Maintenance and hardware failures are handled by the cloud provider
    and are not explicitly modeled.
    """

    def __init__(self, config: Optional[AvailabilityConfig] = None):
        """
        Initialize availability model.

        Args:
            config: Availability configuration. If None, uses defaults.
        """
        self.config = config or DEFAULT_AVAILABILITY_CONFIG

    def get_customer_demand_rate(self, current_time: float) -> float:
        """
        Get time-varying customer demand rate λ_customer(t).

        Models higher demand during business hours.

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            Current customer demand rate (transitions per second)
        """
        hour_of_day = (current_time / 3600) % 24

        if self.config.peak_hours_start <= hour_of_day <= self.config.peak_hours_end:
            # Peak hours
            return self.config.lambda_customer * self.config.peak_demand_multiplier
        else:
            # Off-peak hours
            return self.config.lambda_customer * self.config.off_peak_multiplier

    def get_unavailable_rate(self, current_time: float) -> float:
        """
        Get total rate of transitioning from AVAILABLE to UNAVAILABLE.

        λ_total(t) = λ_customer(t)

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            Total unavailable transition rate (per second)
        """
        return self.get_customer_demand_rate(current_time)

    def get_available_rate(self) -> float:
        """
        Get total rate of transitioning from UNAVAILABLE to AVAILABLE.

        μ_total = μ_release

        Returns:
            Total available transition rate (per second)
        """
        return self.config.mu_release

    def sample_time_to_unavailable(self, current_time: float) -> float:
        """
        Sample time until next UNAVAILABLE event from exponential distribution.

        Time ~ Exponential(λ_total(t))

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            Time until unavailable event (seconds)
        """
        rate = self.get_unavailable_rate(current_time)

        if rate <= 0:
            return float('inf')  # Never transition

        return np.random.exponential(1.0 / rate)

    def sample_time_to_available(self) -> float:
        """
        Sample time until next AVAILABLE event from exponential distribution.

        Time ~ Exponential(μ_total)

        Returns:
            Time until available event (seconds)
        """
        rate = self.get_available_rate()

        if rate <= 0:
            return float('inf')  # Never transition

        return np.random.exponential(1.0 / rate)

    def get_steady_state_availability(self) -> float:
        """
        Calculate steady-state availability (fraction of time AVAILABLE).

        From CTMC theory:
          π_available = μ_release / (λ_customer + μ_release)

        Uses time-averaged rates.

        Returns:
            Steady-state availability probability (0 to 1)
        """
        # Use average customer demand rate
        avg_customer_rate = (
            self.config.lambda_customer *
            (self.config.peak_demand_multiplier + self.config.off_peak_multiplier) / 2
        )

        lambda_total = avg_customer_rate
        mu_total = self.config.mu_release

        return mu_total / (lambda_total + mu_total)

    def get_unavailability_reason(self, current_time: float) -> str:
        """
        Get the reason for becoming unavailable.

        In this simplified model, it's always "customer" (another customer renting the GPU).

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            Reason string: "customer"
        """
        return "customer"

    def get_recovery_reason(self) -> str:
        """
        Get the reason for becoming available.

        In this simplified model, it's always "release" (customer finished renting).

        Returns:
            Reason string: "release"
        """
        return "release"


class TimeVaryingAvailability(AvailabilityModel):
    """
    Extended availability model with custom time-varying functions.

    Allows users to specify custom functions for:
      - λ_customer(t): Time-varying customer demand
      - μ_release(t): Time-varying release rate
    """

    def __init__(
        self,
        config: Optional[AvailabilityConfig] = None,
        customer_demand_fn: Optional[Callable[[float], float]] = None,
        release_rate_fn: Optional[Callable[[float], float]] = None
    ):
        """
        Initialize time-varying availability model.

        Args:
            config: Availability configuration
            customer_demand_fn: Custom λ_customer(t) function
            release_rate_fn: Custom μ_release(t) function
        """
        super().__init__(config)

        self.customer_demand_fn = customer_demand_fn
        self.release_rate_fn = release_rate_fn

    def get_customer_demand_rate(self, current_time: float) -> float:
        """Get customer demand rate, using custom function if provided."""
        if self.customer_demand_fn is not None:
            return self.customer_demand_fn(current_time)

        return super().get_customer_demand_rate(current_time)

    def get_available_rate(self, current_time: float = 0.0) -> float:
        """Get available rate, using custom function if provided."""
        if self.release_rate_fn is not None:
            return self.release_rate_fn(current_time)

        return super().get_available_rate()


# Convenience functions

def get_customer_demand_rate(
    current_time: float,
    base_rate: float = 0.02,
    peak_hours: tuple = (9, 17),
    peak_multiplier: float = 1.5,
    off_peak_multiplier: float = 0.7
) -> float:
    """
    Calculate time-varying customer demand rate.

    Args:
        current_time: Current time (seconds)
        base_rate: Base demand rate (per second)
        peak_hours: (start_hour, end_hour) for peak demand
        peak_multiplier: Demand multiplier during peak hours
        off_peak_multiplier: Demand multiplier off-peak

    Returns:
        Current customer demand rate (per second)
    """
    hour_of_day = (current_time / 3600) % 24

    if peak_hours[0] <= hour_of_day <= peak_hours[1]:
        return base_rate * peak_multiplier
    else:
        return base_rate * off_peak_multiplier


def get_release_rate(
    current_time: float,
    base_rate: float = 0.02,
    time_varying: bool = False
) -> float:
    """
    Calculate resource release rate.

    Args:
        current_time: Current time (seconds)
        base_rate: Base release rate (per second)
        time_varying: Whether release rate varies with time

    Returns:
        Current release rate (per second)
    """
    if not time_varying:
        return base_rate

    # Simple model: higher release rate during off-peak hours
    hour_of_day = (current_time / 3600) % 24

    if 9 <= hour_of_day <= 17:
        return base_rate * 0.8  # Slower release during peak
    else:
        return base_rate * 1.2  # Faster release off-peak


if __name__ == "__main__":
    # Example usage
    print("CGRSP Customer-Only Availability Models")
    print("=" * 50)

    # Create availability model
    model = AvailabilityModel()

    # Display rates over 24 hours
    print("\nRates over 24 hours:")
    print("-" * 40)
    print(f"{'Hour':<6} {'Customer Rate':<16} {'Unavailable Rate':<16} {'Available Rate':<16}")
    print("-" * 60)

    for hour in range(0, 24, 2):
        time_sec = hour * 3600
        customer_rate = model.get_customer_demand_rate(time_sec)
        unavailable_rate = model.get_unavailable_rate(time_sec)
        available_rate = model.get_available_rate()

        print(f"{hour:<6} {customer_rate:<16.6f} {unavailable_rate:<16.6f} {available_rate:<16.6f}")

    # Display steady-state availability
    print(f"\nSteady-state availability: {model.get_steady_state_availability()*100:.2f}%")

    # Sample transition times
    print(f"\nSample transition times (starting at t=0):")
    print("-" * 40)

    np.random.seed(42)
    time_to_unavailable = model.sample_time_to_unavailable(0.0)
    time_to_available = model.sample_time_to_available()

    print(f"Time to unavailable: {time_to_unavailable:.1f}s ({time_to_unavailable/3600:.2f}h)")
    print(f"Time to available: {time_to_available:.1f}s ({time_to_available/3600:.2f}h)")

    # Test time-varying availability
    print("\n\nTime-Varying Availability Example:")
    print("-" * 40)

    def custom_demand(t: float) -> float:
        """Custom demand with morning and evening peaks."""
        hour = (t / 3600) % 24
        if 7 <= hour <= 9:  # Morning peak
            return 0.01 * 2.0
        elif 17 <= hour <= 19:  # Evening peak
            return 0.01 * 1.8
        else:
            return 0.01 * 0.5

    tv_model = TimeVaryingAvailability(customer_demand_fn=custom_demand)

    print(f"{'Hour':<6} {'Custom Demand':<16}")
    print("-" * 25)
    for hour in range(0, 24, 2):
        time_sec = hour * 3600
        demand = tv_model.get_customer_demand_rate(time_sec)
        print(f"{hour:<6} {demand:<16.6f}")
