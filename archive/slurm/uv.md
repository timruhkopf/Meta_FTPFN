# Use UV on slurm: 

## Install UV
on the login node install UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
```

## Redirect storage to $BIGWORK

You need to tell uv to look at $BIGWORK for its three main storage consumers: the Cache, the Python Toolchain, and the Installed Tools.

Add these lines to your ~/.bashrc (or ~/.bash_profile) so they persist in every session and Slurm job:

```bash
# Create a dedicated uv directory in your BIGWORK area
mkdir -p $BIGWORK/uv_storage

# Redirect uv's main storage locations
export UV_CACHE_DIR="$BIGWORK/uv_storage/cache"
export UV_PYTHON_INSTALL_DIR="$BIGWORK/uv_storage/python"
export UV_TOOL_DIR="$BIGWORK/uv_storage/tools"
```

After adding these, run `source ~/.bashrc`. You can verify it's working by running uv cache dir. 
It should now point to a path starting with /bigwork/.

## Usage in Slurm Scripts
When submitting a job with sbatch, ensure your environment variables are passed or loaded. A typical Slurm script on LUIS would look like this:

```Bash

#!/bin/bash
#SBATCH --partition=gpu          # or 'all', 'cpu', etc.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=uv_project


# Ensure uv is in PATH if not already
export PATH="$HOME/.cargo/bin:$PATH"

# Recommended: Move your project directory to $BIGWORK as well
cd $BIGWORK/my_project

# Create a venv (stored locally in the project folder on $BIGWORK)
uv venv
source .venv/bin/activate

# Install and run
uv pip install -r requirements.txt
uv run --frozen my_script.py # this is faster, but assumes your .venv is in sync with uv.lock
```



4. Environment Drift (uv --frozen)
Using uv run --frozen is great for speed, but it assumes your .venv is perfectly in sync with your uv.lock.

The Risk: If you modify your pyproject.toml on the login node but forget to run uv lock or uv sync before submitting the 100-job array, the jobs will run with the old environment.

Recommendation: Always run uv sync once before submitting a large batch of jobs.