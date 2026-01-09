source: https://github.com/violincoding/mlflow-setup with-postgresql-and-minio
 
3. Stripped-down Docker Setup (Head Node)
You want to keep the PostgreSQL database and the MLflow server,
2. but remove MinIO. Since you aren't using remote artifact storage,
3. MLflow will default to storing files (like .pkl models) in a local folder.

Note: If you are on a Slurm cluster, "local" folder usually means a 
shared network drive (NFS/Lustre) so that all jobs can see the same files.

Updated `docker-compose.yaml`
```YAML

version: "3.8"

services:
  db:
    image: postgres:latest
    container_name: mlflow_db
    restart: always
    environment:
      # TODO setup an env file with these values!
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlflow_db
    ports:
      # Expose Postgres port to host (outside of docker) for external communication
      - "5432:5432"
    volumes:
      # persistent database storage (for when the job terminates) 
      # communicating outside of docker
      - ./postgres_data:/var/lib/postgresql/data

  mlflow:
    # Use the official MLflow image
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    restart: always
    ports: # Expose MLflow server port to host (outside of docker)
      - "5000:5000"
    environment:
      - # TODO setup an env file with these values!
      - BACKEND_STORE_URI=postgresql://mlflow:password@db:5432/mlflow_db
    # Finally, set the command to start the MLflow server in the container
    command: >
      mlflow server 
      --backend-store-uri postgresql://mlflow:password@db:5432/mlflow_db
      --default-artifact-root /shared/path/to/mlruns
      --host 0.0.0.0
    depends_on: # Ensure MLflow starts after the database is ready
      - db
        
```


2. Communicating from Slurm Jobs
In a cluster, your compute nodes (where the experiment runs) need the IP address 
3. or Hostname of the head node where Docker is running.

Step A: Get the Head Node IP
Run this on the head node where you started Docker:


```bash
hostname -I | awk '{print $1}'
```
# Example Output: 192.168.1.50
Step B: The Slurm Submit Script
When you submit your experiment job, you must tell MLflow where the server is 
using the `MLFLOW_TRACKING_URI` environment variable.


``` bash
#!/bin/bash
#SBATCH --job-name=mlflow_experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
```
# 1. Set the tracking URI to your Head Node's IP
`export MLFLOW_TRACKING_URI="http://192.168.1.50:5000"`

# 2. Run your script
`python train.py`

3. Critical Configuration Changes
To ensure this works without MinIO and on Slurm, keep these three points in mind:

Artifact Root: In the command section of the `docker-compose.yaml`, 
the `--default-artifact-root` must point to a directory that both the Docker
container and the Slurm compute nodes can access. Usually, this is a path on
your cluster's shared scratch or home space (e.g., /home/user/mlruns).

Database Driver: Your Slurm environment (where your Python script runs) 
needs the PostgreSQL driver installed.


```Bash
pip install mlflow psycopg2-binary
```
The "Experiment Job" Logic: In your `train.py`, you don't need to do anything special.
As long as the environment variable is set, MLflow automatically picks it up:

```Python
import mlflow
import os 
# No need for set_tracking_uri if MLFLOW_TRACKING_URI is in env
with mlflow.start_run():
    mlflow.log_param("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
    mlflow.log_metric("accuracy", 0.99)
```