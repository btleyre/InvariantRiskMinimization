import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths",
                        nargs='+',
                        help="The paths to folders containing csvs.")
    parser.add_argument('--input_names',
                        nargs='+',
                        help='The names for the corresponding models.')
    parser.add_argument('--save_path',
                        default=None,
                        help='the path to save to')        
    args = parser.parse_args()

    name_array = np.array(args.input_names)
    print(name_array)

    all_frames = []
    # First, load in all the CSVs as dataframes.
    print(args.input_paths)
    for csv_path in args.input_paths:
        print(csv_path)
        all_files = glob.glob(csv_path + "**/*.csv")
        print(all_files)
        for full_path in all_files:
            all_frames.append(
                pd.read_csv(full_path)
            )

    print(all_frames)

    compiled = pd.concat(all_frames)

    compiled.to_csv(args.save_path)


if __name__ == "__main__":
    main()
