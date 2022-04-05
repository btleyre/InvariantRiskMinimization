"""Gridsearch for nli experiments"""
import argparse
import os

import subprocess


def main():

    # All experiments

    for hid_dim in [50, 250, 400]:
        for lr in [5e-5, 5e-3]:
            for l2 in [0.01, 0.00001]:
                for penalty_weight in [0.1, 1.0, 1000.0]:
                    for sigma in [0.01, 8.0]:
                        job_name = "{}_hid_{}_lr_{}_l2_{}_pen_{}_sigma".format(
                            hid_dim,
                            lr,
                            lr,
                            penalty_weight,
                            sigma
                        )

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/grid_invar_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/invar_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/grid_invar_{}/'.format(job_name),
                            '--invar_penalty'

                        ]
                        subprocess.run(cmd)

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/grid_new_invar_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/new_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/grid_new_invar_{}/'.format(job_name),
                            '--new_invar_penalty',
                        ]
                        subprocess.run(cmd)

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/grid_both_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/both_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/grid_both_{}/'.format(job_name),
                            '--new_invar_penalty',
                            '--invar_penalty'
                        ]
                        subprocess.run(cmd)
                        1/0


if __name__ == "__main__":
    main()
