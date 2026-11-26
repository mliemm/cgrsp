"""
CGRSP Evaluation Framework

Provides metrics calculation, experiment running, sensitivity analysis,
verification testing, visualization, and report generation capabilities.
"""

from cgrsp.evaluation.metrics import (
    PerformanceMetrics,
    QoSMetrics,
    OperationalMetrics,
    StatisticalAnalysis,
    ObjectiveFunctionDecomposition,
)

from cgrsp.evaluation.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
)

from cgrsp.evaluation.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    ParameterSweep,
)

from cgrsp.evaluation.verification import (
    ConstraintVerifier,
    DeadlineVerifier,
    AssignmentVerifier,
    CostVerifier,
    VerificationReport,
)

from cgrsp.evaluation.visualization import (
    PerformanceVisualizer,
    TimeSeriesVisualizer,
    SensitivityVisualizer,
)

from cgrsp.evaluation.reports import (
    ReportGenerator,
    ExperimentReport,
)

__all__ = [
    # Metrics
    "PerformanceMetrics",
    "QoSMetrics",
    "OperationalMetrics",
    "StatisticalAnalysis",
    "ObjectiveFunctionDecomposition",
    # Runner
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    # Sensitivity
    "SensitivityAnalyzer",
    "SensitivityResult",
    "ParameterSweep",
    # Verification
    "ConstraintVerifier",
    "DeadlineVerifier",
    "AssignmentVerifier",
    "CostVerifier",
    "VerificationReport",
    # Visualization
    "PerformanceVisualizer",
    "TimeSeriesVisualizer",
    "SensitivityVisualizer",
    # Reports
    "ReportGenerator",
    "ExperimentReport",
]
