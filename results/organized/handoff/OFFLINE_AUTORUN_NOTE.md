# Offline / autorun note

## What is now supported

The repository now includes a higher-level orchestration script:

- `orchestrate_response_supervision_experiment.py`

Its purpose is to make response-supervision reruns more reliable by:
- launching both required branches itself
- recording the real PIDs it launched
- using `caffeinate -dimsu` when available to help prevent sleep
- waiting for the **full partial-best file set** instead of only a metadata JSON
- refreshing downstream figures only after analyzable partial-best assets exist

## What it watches for

For each branch, it waits for all of:
- `best_config.partial.json`
- `best_model_params.partial.npz`
- `best_test_predictions.partial.npz`
- `best_checkpoint_meta.json`

Only then does it trigger downstream figure refresh.

## Important limitation

This improves robustness, but it still cannot guarantee continued execution if the machine is forcibly put to sleep or shut down. In particular:

- if the laptop lid is closed and the system sleeps, local training may pause
- `caffeinate` helps only while the machine remains powered and awake

## Practical recommendation

If you want the most reliable unattended local run:
1. keep the machine plugged in
2. do not close the lid
3. use the orchestrator rather than manually launching training and a separate monitor

## Why this matters

Earlier monitoring logic only watched for metadata JSON creation and could declare success too early. The orchestrator prevents that by requiring the full partial-best artifact set before downstream analysis is refreshed.
