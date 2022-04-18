"""Gridsearch for nli experiments"""
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
                            l2,
                            penalty_weight,
                            sigma
                        )

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/sphere_grid_invar_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/sphere_invar_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/sphere_grid_invar_{}/'.format(job_name),
                            '--invar_penalty',
                            '--use_reg'

                        ]
                        subprocess.run(cmd)

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/sphere_grid_new_invar_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/sphere_new_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/sphere_grid_new_invar_{}/'.format(job_name),
                            '--new_invar_penalty',
                            '--use_reg'
                        ]
                        subprocess.run(cmd)

                        # Create a directory to house the artifacts.
                        os.mkdir('./grid_output/sphere_grid_both_{}'.format(job_name))

                        cmd = [
                            'sbatch',
                            '--partition=gpu',
                            '--output=./grid_output/sphere_both_{}_%j.out'.format(job_name),
                            '--job-name={}'.format(job_name),
                            'grid_job',
                            str(hid_dim),
                            str(l2),
                            str(lr),
                            str(penalty_weight),
                            str(sigma),
                            './grid_output/sphere_grid_both_{}/'.format(job_name),
                            '--new_invar_penalty',
                            '--invar_penalty',
                            '--use_reg'
                        ]
                        subprocess.run(cmd)


if __name__ == "__main__":
    main()
