import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument(
        "--fileName",
        required=True,
        type=str,
        help="path to the analysis file (json)"
    )



    # read all settings into one variable and check setting validity
    args = parser.parse_args()
    with open(args.fileName, 'r') as f:
        f.readline()
        # print(f.readline())
        lsLines = [json.loads(s)["status"] for s in f.readlines()]
    print(Counter(lsLines).items())



main()