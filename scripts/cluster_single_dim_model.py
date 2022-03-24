import argparse
import logging
import os
from multiprocessing import Pool

log = logging.getLogger("ClusterRunner")

parser = argparse.ArgumentParser(description="Run soccer script")
parser.add_argument(
    "-i", "--id", type=int, help="ID", required=True,
)


args = parser.parse_args()
args.id = args.id - 1
log.info(f"ID: {args.id}")

begin_end = [("2021-07-01", "2021-12-01")]
age_groups = [
    "0-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90+",
]


mapping = []

for be in begin_end:
    for a in age_groups:
        ma = []
        ma.append(be)
        ma.append(a)
        mapping.append(tuple(ma))


num_jobs_per_node = 9
mapping_clustered = []
ended = False
for i in range(len(mapping)):
    if not num_jobs_per_node * i >= len(mapping):
        mapping_clustered.append([])
    for j in range(num_jobs_per_node):
        i_mapping = num_jobs_per_node * i + j
        if i_mapping < len(mapping):
            mapping_clustered[-1].append(mapping[i_mapping])
        else:
            ended = True
            break
    if ended:
        break


def exec(args_list):
    """
    Executes python script
    """
    (begin_end, age_group,) = args_list
    os.system(
        f"python infer_single_age_group.py "
        f"-b {begin_end[0]} -e {begin_end[1]} "
        f"-a {age_group} "
    )


if __name__ == "__main__":
    with Pool(num_jobs_per_node) as p:
        p.map(exec, mapping_clustered[args.id])
