import mlflow
import multiprocessing
import datetime
import os
import sqlite3
import time
import random

DB_PATH = "minimal_stress.db"
NUM_WORKERS = 10  # Start with 10 to see the "queueing" effect
STEPS = 1000


def initialize_db():
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    # 1. Create schema
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    mlflow.search_experiments()
    # 2. Set WAL mode (Essential for any concurrency)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()


def worker(worker_id, start_time_iso):
    # The 'timeout' is the only "logic" we use—it's a standard SQLite param
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}?timeout=60")

    # Synchronize all processes to the exact same start time
    start_t = datetime.datetime.fromisoformat(start_time_iso)
    while datetime.datetime.now() < start_t:
        pass

    try:
        with mlflow.start_run(run_name=f"Worker_{worker_id}"):
            for step in range(STEPS):
                mlflow.log_metric("val", worker_id, step=step)
                time.sleep(random.uniform(0.1, 0.5))
        print(f"Worker {worker_id}: Success")
    except Exception as e:
        print(f"Worker {worker_id}: FAILED -> {e}")


if __name__ == "__main__":
    initialize_db()

    # Start in 3 seconds
    barrier = (datetime.datetime.now() + datetime.timedelta(seconds=3)).isoformat()

    procs = [multiprocessing.Process(target=worker, args=(i, barrier)) for i in range(NUM_WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()

    # Final Audit
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    runs = mlflow.search_runs()
    print(f"\n--- RESULTS ---")
    print(f"Successful Jobs: {len(runs[runs['status'] == 'FINISHED'])} / {NUM_WORKERS}")