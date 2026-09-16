#!/bin/bash

ssh -tt -L 5000:127.0.0.1:5000 ulysses '
    cd ~/PycharmProjects/Meta_FTPFN &&
    MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui \
      --backend-store-uri file://$HOME/PycharmProjects/Meta_FTPFN/mlruns \
      --host 127.0.0.1 --port 5000
'
