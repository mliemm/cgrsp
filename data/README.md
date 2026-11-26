# CGRSP Research Data Repository

Unified directory containing datasets, collection scripts, and analysis tools for the **Cloud GPU Rendering Scheduling Problem (CGRSP)** research.

## 📁 Directory Structure

```
data/
├── README.md                      # This file
├── .env                           # Environment variables (API keys)
├── runpod_pricing/                # GPU pricing & availability data
│   ├── README.md                  # Dataset documentation
│   ├── distributions_*.json       # Statistical summaries
│   ├── history_*.json             # Raw GPU instance data
│   └── snapshot_*.json            # Time-series snapshots
├── rendering_jobs/                # Production rendering job data
│   ├── README.md                  # Dataset documentation
│   ├── *-request.json             # Job input parameters
│   └── *-response.json            # Job execution results
├── scripts/                       # Data collection & analysis
│   ├── runpod_monitor.py          # GPU pricing monitor
│   └── analyze_data.py            # Data analysis tools
├── archive/                       # Historical scripts & configs
├── unused_after_filter/           # Unfiltered, excluded and superseded data (not read by any code)
├── raw_rendering_responses.jsonl  # Raw response data (not distributed: carries credentials)
└── rendering_performance_data.csv # Performance metrics
```

Only `rendering_performance_data.csv`, the `unused_after_filter/` folder and this README are tracked in the repository. The `.env` file, `scripts/`, `runpod_pricing/`, `rendering_jobs/`, `archive/` and the raw response file exist only on the author's machine, so the Quick Start below does not run from a fresh clone; the thesis results need only the tracked CSV.

## 🚀 Quick Start

### GPU Monitoring
```bash
cd data/
python scripts/runpod_monitor.py
```

### Data Analysis
```bash
cd data/
python scripts/analyze_data.py
```

## ⚙️ Configuration

Set up your `.env` file:
```
RUNPOD_API_KEY=your_api_key_here
```

## 📊 Datasets

### 1. RunPod GPU Pricing & Availability
- **Period:** March 11 - April 20, 2026 (40 days)
- **Files:** 5,407 JSON files (5,294 snapshots, 77 distribution summaries, 36 history files)
- **GPU Models:** 43 unique types
- **Purpose:** Real-world GPU marketplace dynamics

### 2. Rendering Jobs Workload
- **Period:** July 2025 - June 2026
- **Jobs:** 6,627 request/response pairs in `unused_after_filter/rendering_performance_data_unfiltered.csv`; `rendering_performance_data.csv` keeps the 6,234 that never ran while more than five jobs were in flight (RunPod's default of five workers), and `unused_after_filter/rendering_performance_excluded_over5.csv` lists the 393 removed rows; the same folder keeps the superseded August 2025 consolidation files (`combined_rendering_data*.jsonl`, `rendering_performance_data_original.csv`). See its README. `scripts/filter_over_five_concurrency.py` (repository root) regenerates both from the unfiltered file.
- **Purpose:** Production GPU rendering characterization
- **Identifiers:** the CSV keeps numeric `material_id` and `model_id` values from the author's pipeline; it contains no client names, account identifiers, or billing details.
- **Cost columns:** cost_per_second, cost_per_hour, total_cost, render_cost and cost_per_megapixel in rendering_performance_data.csv are recomputed at the RunPod Pods securePrice the simulator uses (11 March 2026: $0.46/$0.27/$0.25/$0.25 per hour for RTX 3090/A5000/A4500/A4000), not the prices production paid. Production ran on a RunPod Serverless endpoint; the raw responses carry no cost, and `unused_after_filter/combined_rendering_data.jsonl` prices its August 2025 records at the Serverless Flex rates ($0.684/hr for A5000 and RTX 3090, $0.576/hr for A4000 and A4500). No analysis code reads these columns; the execution-time model uses render_time_s only.

## 🛠️ Scripts

### Active Scripts (`scripts/`)
- **runpod_monitor.py** - Automated GPU pricing collection
- **analyze_data.py** - Comprehensive data analysis

### Archived Scripts (`archive/`)
Historical scripts and configurations preserved for reference.

## 📈 Research Applications

These datasets support CGRSP scheduler development:
- **Realistic workloads** from production data
- **GPU availability patterns** from marketplace monitoring
- **Performance baselines** for scheduler evaluation
- **Cost optimization** based on real pricing data

## 🔗 Integration with CGRSP

```python
# Use datasets in CGRSP simulator
from data.runpod_pricing import load_gpu_availability
from data.rendering_jobs import load_job_trace

# Create realistic test scenarios
scenario = create_cgrsp_scenario(
    gpu_availability=load_gpu_availability(),
    workload_trace=load_job_trace()
)
```

---

**Last Updated:** April 15, 2026
**Status:** ✅ Unified Data Repository
**Research:** CGRSP Scheduler Development