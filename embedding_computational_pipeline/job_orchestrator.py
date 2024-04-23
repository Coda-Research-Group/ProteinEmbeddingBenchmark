import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml
from jinja2 import Environment, FileSystemLoader
from kubernetes import client, config

"""
This script is used to orchestrate the jobs to generate embeddings for the proteins.
The script will generate a job for each protein in the range [from_job_id, from_job_id + num_jobs].

It takes four arguments:
    --job_template (-t): The path to the jinja2 template file. Required.
    --output_log (-o): The path to the output file where the results are written. Default: job_log.txt
    --start_id (-i): The job id to start from. Required.
    --jobs_number (-j): The number of jobs to run. Required.
    --namespace (-n): The namespace to run the jobs in. Required.
    --max_concurrent_jobs (-m): The maximum number of concurrent jobs. Default: 20
    --dry-run (-d): If set, the jobs will not be submitted to the cluster. Default: False

Example usage:
    python job_orchestrator.py -t job_templates/grasr-job.yaml.jinja2 -i 1 -j 2 -n fi-lmi-ns

Requirements:
    - Kubeconfig file in ~/.kube/config with access to the namespace
    - and requirements in requirements.txt installed
"""


def run_job(job):
    # Load the kubeconfig file ~/.kube/config which contains the credentials to access the cluster
    # and in particular the namespace defined in the argument
    config.load_kube_config(config_file="kuba-cluster.yaml")
    api_instance = client.BatchV1Api()
    manifest = yaml.safe_load(job.get("manifest"))
    api_job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=manifest.get("metadata"),
        spec=manifest.get("spec"),
    )

    # If dry run is set, just print the job and return after a random time
    if args.dry_run:
        print(f"Job {job.get('job_id')}: dry run")
        time.sleep(random.randint(1, 3) * 10)
        return job

    # Check if the job already exists and create it if it doesn't
    try:
        api_instance.read_namespaced_job_status(
            name=manifest.get("metadata").get("name"), namespace=args.namespace
        ).status
        print(f"Job {job.get('job_id')}: exists")
    except client.ApiException as e:
        print(f"Job {job.get('job_id')}: creating")
        # Run the job
        api_instance.create_namespaced_job(body=api_job, namespace=args.namespace)

    # Wait for the job to complete and update the status
    while True:
        job_status = api_instance.read_namespaced_job_status(
            name=manifest.get("metadata").get("name"), namespace=args.namespace
        )

        if job_status.status.succeeded == 1:
            job.update({"status": "completed"})
            print(f"Job {job.get('job_id')}: completed")
            return job
        elif job_status.status.failed == 1:
            job.update({"status": "failed"})
            print(f"Job {job.get('job_id')}: failed")
            return job
        else:
            job.update({"status": "running"})
            # print(f"Job {job.get('job_id')}: running")

        # Wait for 60 seconds before checking the status of the job in kubernetes again
        time.sleep(60)


def orchestrate_jobs(args):
    num_jobs = args.jobs_number
    from_job_id = args.start_id
    max_concurrent_jobs = args.max_concurrent_jobs
    completed_jobs = {}
    failed_jobs = {}

    template_path = args.job_template
    output_file = args.output_log

    if os.path.exists(output_file):
        os.remove(output_file)

    j2_env = Environment(loader=FileSystemLoader("."))

    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        futures = []
        job_template_jinja2 = j2_env.get_template(template_path)

        job_id = from_job_id
        wait_time = 0
        # Loop over the jobs and submit them to the cluster
        while job_id <= from_job_id + num_jobs - 1:
            # Wait until there are less than max_concurrent_jobs jobs running
            if executor._work_queue.qsize() >= args.max_concurrent_jobs:
                print(
                    f"Waiting for jobs to complete. Jobs todo: {executor._work_queue.qsize()}, wait time: {2**wait_time}"
                )
                time.sleep(2**wait_time)
                wait_time += 1
                continue
            wait_time = 0
            job_name = f"job-cif2emb-{job_id}"
            print(f"Submitting job {job_name}")

            # Render the jinja2 template of the job
            rendered_template = job_template_jinja2.render(
                job_iterator=job_id,
                job_name=job_name,
            )
            # Create a job definition for the executor
            job_definition = {
                "job_id": job_id,
                "manifest": rendered_template,
            }
            # Submit the job to the executor
            future = executor.submit(run_job, job_definition)
            futures.append(future)
            job_id += 1

            time.sleep(2)

        # Wait for all jobs to complete
        print("Waiting for all jobs to complete.")
        for future in futures:
            job_result = future.result()
            job_id = int(job_result.get("job_id"))
            if job_result.get("status") == "failed":
                failed_jobs[job_id] = job_result.get("job_id")
            elif job_result.get("status") == "compleCompleted jobsted":
                completed_jobs[job_id] = job_result.get("job_id")

        print("All jobs completed.")

    """ Write the results to the output file in the format:
        Completed jobs:
        job_id
        ...
    
        Failed jobs:
        job_id
        ...
    """
    with open(output_file, "a") as f:
        f.write("Completed jobs:\n")
        for job_id, result in completed_jobs.items():
            f.write(f"{result}\n")

        f.write("\nFailed jobs:\n")
        for job_id, result in failed_jobs.items():
            f.write(f"{result}\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--job_template", "-t", help="Path to the jinja2 template file.", required=True
    )
    argparser.add_argument(
        "--output_log",
        "-o",
        help="Path to the output file where the results are written.",
        default="job_log.txt",
    )
    argparser.add_argument(
        "--start_id", "-i", help="The job id to start from.", type=int, required=True
    )
    argparser.add_argument(
        "--jobs_number",
        "-j",
        help="The number of jobs to run.",
        type=int,
        required=True,
    )
    argparser.add_argument(
        "--namespace", "-n", help="The namespace to run the jobs in", required=True
    )
    argparser.add_argument(
        "--max_concurrent_jobs",
        "-m",
        help="The maximum number of concurrent jobs",
        type=int,
        default=20,
    )
    argparser.add_argument(
        "--dry-run",
        "-d",
        help="If set, the jobs will not be submitted to the cluster",
        action="store_true",
    )
    args = argparser.parse_args()

    orchestrate_jobs(args)
