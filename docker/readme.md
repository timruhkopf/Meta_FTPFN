
# PostgreSQL + MLFlow Server setup on Ulysses (remote desktop)

## 1. Install Docker

Follow the steps here: [docker / ubuntu install](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

## 2. Docker compose setup
Create a `docker-compose.yaml` file with the services for PostgreSQL and MLflow (notice, that docker/mlflow 
is set up with a Dockerfile and requirements for mlflow to run with postgres), run 
the following command to start the services:

```bash
cd docker # the directory where docker-compose.yaml is located
docker-compose up -d --build
``` 

We can verify that the services are running using:

```bash
#ruhkopf@ulysses:~$ 
docker ps 

netstat -tulpn | grep 5001
#(Not all processes could be identified, non-owned process info
# will not be shown, you would have to be root to see it all.)
#tcp        0      0 0.0.0.0:5001            0.0.0.0:*               LISTEN      -                   
#tcp6       0      0 :::5001                 :::*                    LISTEN      -   
```

In case we want to stop the services, we can run:

```bash
docker-compose down
```
during dev, we may also want to clean up volumes:

```bash
docker-compose down -v
```


## 3. Ulysses writes mlflow to localhost 
To use mlflow with postgresql db in the python script, we need to set the tracking uri to the localhost. 
Since we are running the mlflow server on a docker container on ulysses and have exposed the docker's 
port 5001 to ulysses' port 5001, we can access the mlflow server on ulysses via localhost:5001

```python
import mlflow
# mlflow.set_tracking_uri("postgresql://mlflow:password@localhost:5001")
mlflow.set_tracking_uri("http://localhost:5001") 
```


## 3. From local listen to ulysses: 
On the local machine, run the following command to create an SSH tunnel that forwards
the local port 5002  to the remote server's port 5001 and keep the session running:

```bash
ssh -J ruhkopf@ssh1.ai.uni-hannover.de -L 5002:localhost:5001 ruhkopf@130.75.145.188
```

Notice, that when the docker container is still running after a connection loss, we can just reconnect 
with the same command again, loging in with the password during the cmd verification step. 
check with `docker ps` on ulysses, that the container is still running in advance.

Open your browser and navigate to `http://localhost:5002` to access the MLflow UI.









source: https://github.com/violincoding/mlflow-setup with-postgresql-and-minio

[//]: # ()
[//]: # (# Untested: )

[//]: # ()
[//]: # (3. Stripped-down Docker Setup &#40;Head Node&#41;)

[//]: # (You want to keep the PostgreSQL database and the MLflow server,)

[//]: # (2. but remove MinIO. Since you aren't using remote artifact storage,)

[//]: # (3. MLflow will default to storing files &#40;like .pkl models&#41; in a local folder.)

[//]: # ()
[//]: # (Note: If you are on a Slurm cluster, "local" folder usually means a )

[//]: # (shared network drive &#40;NFS/Lustre&#41; so that all jobs can see the same files.)

[//]: # ()
[//]: # (Updated `docker-compose.yaml`)

[//]: # (```YAML)

[//]: # ()
[//]: # (version: "3.8")

[//]: # ()
[//]: # (services:)

[//]: # (  db:)

[//]: # (    image: postgres:latest)

[//]: # (    container_name: mlflow_db)

[//]: # (    restart: always)

[//]: # (    environment:)

[//]: # (      # TODO setup an env file with these values!)

[//]: # (      - POSTGRES_USER=mlflow)

[//]: # (      - POSTGRES_PASSWORD=password)

[//]: # (      - POSTGRES_DB=mlflow_db)

[//]: # (    ports:)

[//]: # (      # Expose Postgres port to host &#40;outside of docker&#41; for external communication)

[//]: # (      - "5432:5432")

[//]: # (    volumes:)

[//]: # (      # persistent database storage &#40;for when the job terminates&#41; )

[//]: # (      # communicating outside of docker)

[//]: # (      - ./postgres_data:/var/lib/postgresql/data)

[//]: # ()
[//]: # (  mlflow:)

[//]: # (    # Use the official MLflow image)

[//]: # (    image: ghcr.io/mlflow/mlflow:latest)

[//]: # (    container_name: mlflow_server)

[//]: # (    restart: always)

[//]: # (    ports: # Expose MLflow server port to host &#40;outside of docker&#41;)

[//]: # (      - "5000:5000")

[//]: # (    environment:)

[//]: # (      - # TODO setup an env file with these values!)

[//]: # (      - BACKEND_STORE_URI=postgresql://mlflow:password@db:5432/mlflow_db)

[//]: # (    # Finally, set the command to start the MLflow server in the container)

[//]: # (    command: >)

[//]: # (      mlflow server )

[//]: # (      --backend-store-uri postgresql://mlflow:password@db:5432/mlflow_db)

[//]: # (      --default-artifact-root /shared/path/to/mlruns)

[//]: # (      --host 0.0.0.0)

[//]: # (    depends_on: # Ensure MLflow starts after the database is ready)

[//]: # (      - db)

[//]: # (        )
[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (2. Communicating from Slurm Jobs)

[//]: # (In a cluster, your compute nodes &#40;where the experiment runs&#41; need the IP address )

[//]: # (3. or Hostname of the head node where Docker is running.)

[//]: # ()
[//]: # (Step A: Get the Head Node IP)

[//]: # (Run this on the head node where you started Docker:)

[//]: # ()
[//]: # ()
[//]: # (```bash)

[//]: # (hostname -I | awk '{print $1}')

[//]: # (```)

[//]: # (# Example Output: 192.168.1.50)

[//]: # (Step B: The Slurm Submit Script)

[//]: # (When you submit your experiment job, you must tell MLflow where the server is )

[//]: # (using the `MLFLOW_TRACKING_URI` environment variable.)

[//]: # ()
[//]: # ()
[//]: # (``` bash)

[//]: # (#!/bin/bash)

[//]: # (#SBATCH --job-name=mlflow_experiment)

[//]: # (#SBATCH --nodes=1)

[//]: # (#SBATCH --ntasks=1)

[//]: # (#SBATCH --partition=compute)

[//]: # (```)

[//]: # (# 1. Set the tracking URI to your Head Node's IP)

[//]: # (`export MLFLOW_TRACKING_URI="http://192.168.1.50:5000"`)

[//]: # ()
[//]: # (# 2. Run your script)

[//]: # (`python train.py`)

[//]: # ()
[//]: # (3. Critical Configuration Changes)

[//]: # (To ensure this works without MinIO and on Slurm, keep these three points in mind:)

[//]: # ()
[//]: # (Artifact Root: In the command section of the `docker-compose.yaml`, )

[//]: # (the `--default-artifact-root` must point to a directory that both the Docker)

[//]: # (container and the Slurm compute nodes can access. Usually, this is a path on)

[//]: # (your cluster's shared scratch or home space &#40;e.g., /home/user/mlruns&#41;.)

[//]: # ()
[//]: # (Database Driver: Your Slurm environment &#40;where your Python script runs&#41; )

[//]: # (needs the PostgreSQL driver installed.)

[//]: # ()
[//]: # ()
[//]: # (```Bash)

[//]: # (pip install mlflow psycopg2-binary)

[//]: # (```)

[//]: # (The "Experiment Job" Logic: In your `train.py`, you don't need to do anything special.)

[//]: # (As long as the environment variable is set, MLflow automatically picks it up:)

[//]: # ()
[//]: # (```Python)

[//]: # (import mlflow)

[//]: # (import os )

[//]: # (# No need for set_tracking_uri if MLFLOW_TRACKING_URI is in env)

[//]: # (with mlflow.start_run&#40;&#41;:)

[//]: # (    mlflow.log_param&#40;"slurm_job_id", os.environ.get&#40;"SLURM_JOB_ID"&#41;&#41;)

[//]: # (    mlflow.log_metric&#40;"accuracy", 0.99&#41;)

[//]: # (```)