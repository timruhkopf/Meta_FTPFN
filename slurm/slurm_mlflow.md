# Slurm experiment Tracking with MLflow

First of, we will use the soon to be depreciated local file system tracking of MLFlow, 
meaning that we set `mlflow.set_tracking_uri=file:///<bigwork/user/project>/mlruns` to a local path on the compute node.

With this setup, we can use MLflow to log parameters, metrics, and artifacts during our training runs.

To analyse our results in the dashboard, we will need to do port forwarding from the login node to our local machine.

## on remote
```bash
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "------------------ Starting MLflow UI on port $PORT ------------------"
uv run mlflow ui --backend-store-uri file:///bigwork/nhwpruht/Meta_FTPFN/mlruns --port $PORT
```

## on local 
```bash 
ssh -L <unused-local-port>:localhost:<remote-port> nhwpruht@login.cluster.uni-hannover.de
```
Then open your browser and navigate to `http://localhost:<local-port>` to access the MLflow dashboard.
It may need a refresh to show the data.