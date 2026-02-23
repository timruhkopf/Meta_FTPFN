# Slurm experiment Tracking with MLflow

First of, we will use the soon to be depreciated local file system tracking of MLFlow, 
meaning that we set `mlflow.set_tracking_uri=file:///<bigwork/user/project>/mlruns` to a local path on the compute node.

With this setup, we can use MLflow to log parameters, metrics, and artifacts during our training runs.

To analyse our results in the dashboard, we will need to do port forwarding from the login node to our local machine.

## on remote
```bash
# 1. Get a random free port
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# 2. Identify the specific login node (e.g., login03)
NODE_HOSTNAME=$(hostname)
LOCAL_PORT=5004
USERNAME=nhwpruht

# 3. Construct and echo the command for your local machine
echo "----------------------------------------------------------------"
echo "MLflow UI is starting on $NODE_HOSTNAME:$PORT"
echo "PASTE THIS ON YOUR LOCAL MACHINE:"
echo "ssh -L $LOCAL_PORT:localhost:$PORT $USERNAME@$NODE_HOSTNAME.cluster.uni-hannover.de"
echo "----------------------------------------------------------------"

# 4. Run the UI
uv run mlflow ui --backend-store-uri file:///bigwork/$USERNAME/Meta_FTPFN/mlruns --port $PORT
```

## on local 
```bash 
ssh -L <unused-local-port>:localhost:<remote-port> nhwpruht@login.cluster.uni-hannover.de
```
Then open your browser and navigate to `http://localhost:<local-port>` to access the MLflow dashboard.
It may need a refresh to show the data.






Victory! It’s a common headache with clusters that use multiple login nodes. Here is a concise "cheat sheet" you can save to avoid the "Connection Refused" dance next time.

🚀 MLflow UI on Cluster: Quick Start
Step 1: Start MLflow on a Specific Login Node
Log in to a specific node (e.g., login02) rather than the generic address to ensure your process stays put.

Bash

# 1. Connect to a specific node
ssh nhwpruht@login02.cluster.uni-hannover.de

# 2. Start the UI in the background

```shell
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Running on Port: $PORT"
uv run mlflow ui --host 127.0.0.1 --port $PORT --backend-store-uri file:///bigwork/nhwpruht/Meta_FTPFN/mlruns > mlflow.log 2>&1 &
```

Step 2: Open the Tunnel (Local Machine)
Open a new terminal on your laptop. Match the node number and the port from Step 1.

```Bash

# Template: ssh -L <LocalPort>:127.0.0.1:<RemotePort> user@login<XX>.cluster...
PORT=52801
ssh -L 8080:127.0.0.1:$PORT nhwpruht@login02.cluster.uni-hannover.de
```
Step 3: Access
Navigate to http://localhost:8080 in your browser.

🛠 Troubleshooting Checklist
"Connection Refused"? * Verify you are tunneling to the correct node (login01 vs login02).

Run `ps aux | grep mlflow` on the remote to see if the process is still alive.

"Port already in use"? * Change the first number in the SSH command (e.g., -L 9000:127.0.0.1:...).

Cleanup:

To stop the UI, run `pkill -u nhwpruht -f mlflow` on the cluster.


on local: 
```bash
ps aux | grep "ssh -L 8080"
kill <PID>
```