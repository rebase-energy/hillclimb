#!/usr/bin/env bash
# Re-download the custom-task competition data (CSVs are gitignored).
# Requires Kaggle credentials: set -a; source ../agent-work/.env; set +a
# and having clicked "Join Competition" on kaggle.com for each comp.
set -euo pipefail
cd "$(dirname "$0")"
KAGGLE=../../../mle-bench/.venv/bin/kaggle

DUTCH=dutch-energy-supplier-load-forecasting-challenge
$KAGGLE competitions download -c $DUTCH -p $DUTCH
unzip -o $DUTCH/*.zip -d $DUTCH && rm $DUTCH/*.zip
# normalize names: the harness expects train.csv / test.csv / sample_submission.csv
mv $DUTCH/train_expanded.csv $DUTCH/train.csv
mv $DUTCH/test_new.csv $DUTCH/test.csv
mv $DUTCH/sample_submission_new.csv $DUTCH/sample_submission.csv

ENERGY=energy-forecasting-data-challenge-public
$KAGGLE competitions download -c $ENERGY -p $ENERGY
unzip -o $ENERGY/*.zip -d $ENERGY && rm $ENERGY/*.zip
