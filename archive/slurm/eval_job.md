# Evaluating Model Performance After Training with Slurm Job Dependencies
When training machine learning models using Slurm job arrays, it's common to want to evaluate
the model's performance immediately after training completes, but with another job and different ressource requirements.
This can be efficiently managed using
Slurm's job dependency feature, which allows you to chain jobs together based on the success or

The Slurm-Native Way (Recommended)You can submit your "Analysis" or "Cleanup" job immediately after
your training job, telling Slurm: "Only start this job if Job A finishes successfully."Example
Workflow:

```Bash
# 1. Submit the main training array and capture the Job ID
TRAIN_ID=$(sbatch --parsable train_script.sh)

# 2. Submit the follow-up job (e.g., aggregation or weight syncing)

# afterok means "only run if the previous job exited with code 0"
sbatch --dependency=afterok:$TRAIN_ID aggregation_script.sh
```

Dependency TypesFlagBehaviorafterok:jobidRuns only if the previous job finished successfully (exit
code 0).afterany:jobidRuns regardless of whether the previous job succeeded or failed.afternotok:
jobidRuns only if the previous job failed.

## Important Considerations for Job Arrays
   Since you are using --array=1-100, you have two choices for the follow-up:


1. (PREFERRED) Run once for EACH task in the array: If you want task #5 to trigger its own specific post-processing
   the moment it finishes (without waiting for task #6), you use: 
    `--dependency=afterok:$TRAIN_ID_$_SLURM_ARRAY_TASK_ID`

2. Run once after the ENTIRE array is done: Use `afterok:$TRAIN_ID`. Slurm will wait until every single
   one of the 100 tasks has finished successfully.


CAREFULL with the current setup, a poking with 
`scancel --signal=USR1 <job_id>`, requires the evaluation jobs to handle the signal properly as well, otherwise
they might be left hanging or not started at all. Make sure to test this behavior in a safe environment before deploying it in production.