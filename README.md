## Run

Single commodity:

```powershell
python run_qlib_forecast.py --code CU
```

All configured commodities:

```powershell
python run_qlib_forecast.py --code ALL
```

Selected commodities:

```powershell
python run_qlib_forecast.py --codes CU,AU,SC
```

Useful research options:

```powershell
python run_qlib_forecast.py --code CU --horizon 20 --monthly-macro-lag-days 15
python run_qlib_forecast.py --code ALL --skip-plot
```

Sweep several forecast horizons:

```powershell
python run_horizon_sweep.py --code ALL
python plot_horizon_sweep.py
```

## Outputs

Each commodity writes:

- `output/<code>_qlib_predictions.csv`
- `output/<code>_qlib_metrics.csv`
- `output/<code>_qlib_metrics_by_year.csv`
- `output/<code>_qlib_split_summary.csv`
- `output/<code>_qlib_forecast.png`

When running multiple commodities, the script also writes:

- `output/qlib_metrics_summary.csv`

The horizon sweep writes:

- `output/horizon_sweep_summary.csv`
- `output/horizon_sweep_best.csv`
- `output/horizon_sweep_overview.png`
- `output/horizon_sweep_rmse_by_code.png`
- `output/horizon_sweep_best_predictions.png`

