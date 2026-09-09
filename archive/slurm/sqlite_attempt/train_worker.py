import mlflow
import time
import argparse
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()

    # The critical option we are testing
    # If you want to see it FAIL, change timeout=60 to timeout=0
    mlflow.set_tracking_uri(f"{mlflow.get_tracking_uri()}?timeout=60")

    with mlflow.start_run(run_name=f"Array_Task_{args.task_id}"):

        # --- THE SYNC BARRIER ---
        # All jobs wait until the start of the next full minute
        now = datetime.datetime.now()
        start_time = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)

        print(f"Task {args.task_id}: Waiting for synchronized start at {start_time.strftime('%H:%M:%S')}...")
        while datetime.datetime.now() < start_time:
            time.sleep(0.01)

        # --- THE FLOOD ---
        print(f"Task {args.task_id}: Flooding DB now!")
        for step in range(50):
            # No sleep here! Full speed writes to maximize collisions.
            try:
                mlflow.log_metric("stress_val", args.task_id * step, step=step)
            except Exception as e:
                print(f"Task {args.task_id} FAILED at step {step}: {e}")
                break


if __name__ == "__main__":
    main()