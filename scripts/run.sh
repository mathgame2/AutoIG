#!/bin/bash

# start the instance generation process with irace
# the tuner is chosen by setup.py --tuner (python by default); experiments set up before that option existed use R irace
tuner=$(python3 -c 'import json; print(json.load(open("config.json")).get("tuner", "r"))')
if [ "$tuner" = "r" ]; then
    bash run-irace.sh
else
    python3 -u $AUTOIG/scripts/run_tuning.py
fi

# extract output
pushd detailed-output
bash $AUTOIG/scripts/read-output-files.sh
bash $AUTOIG/scripts/extract-md5sum.sh
popd

# collect results and statistics
python $AUTOIG/scripts/collect_results.py ./
cp detailed-output/detailed-results.json ./
