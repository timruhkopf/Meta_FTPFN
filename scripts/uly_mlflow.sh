#!/bin/bash

# Points at Meta_FTPFN_lupi -- the worktree the id-token/LUPI baseline
# experiments actually run from (2026-09-16; ~/PycharmProjects/Meta_FTPFN
# itself is a separate, stale feature/checkpoint checkout, not used here).
ssh -tt -L 5000:127.0.0.1:5000 ulysses '
    cd ~/PycharmProjects/Meta_FTPFN_lupi &&
    MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui \
      --backend-store-uri file://$HOME/PycharmProjects/Meta_FTPFN_lupi/mlruns \
      --host 127.0.0.1 --port 5000
'
