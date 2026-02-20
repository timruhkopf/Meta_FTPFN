# Setting up SLURM: 

## 1. Setup Instructions
git clone the repo, 

```bash
git clone https://github.com/timruhkopf/Meta_FTPFN.git
uv sync 
uv pip install -e . 
```

then collect both the main branch and the icml-2024 branch of ifBO repository
```bash 
mkdir -p external
cd external
git clone https://github.com/automl/ifBO.git
cd external/ifBO

uv python -m pip install -e external/ifBO
uv python -m pip install -r external/ifBO/core_requirements.txt

cd .. 

# add ifbo_icml2024 branch as folder 
git clone https://github.com/automl/ifBO.git ifbo_icml2024
cd ifbo_icml2024
git fetch origin icml-2024
git checkout icml-2024
```

 
## 2. Prepare the PFN checkpoint

# Collect the ft-pfn checkpoint and move it into `models/pfn_ckpt/.model` 

#TODO: automate this

