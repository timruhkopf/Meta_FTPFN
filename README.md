Meta-FTPFN — local workspace notes

This repository contains local work that depends on the `ifBO` project
(`automl/ifBO`) from the `icml-2024` branch. The `ifBO` code has been cloned
into `external/ifBO` for editable development.

Integration options included here:

- **Editable (recommended for development / patching)**: development clone is
	at `external/ifBO`. Install into your environment with:

	```bash
	# from project root
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

	Or, if you prefer to use a reproducible pinned install (non-editable):

	```bash
	python -m pip install -r requirements-ifbo.txt
	```

- **Using `uv` (recommended workflow you asked for)**:
	- If you use a `uv` virtual environment or task runner, you can run the
		same pip commands inside the `uv` environment. Example patterns:

	```bash
	# create or activate uv-managed env (if configured)
	uv pip install -e external/ifBO
	uv pip install -r external/ifBO/core_requirements.txt
	```

	If you prefer to pin the dependency instead of editable development, use:

	```bash
	uv pip install -r requirements-ifbo.txt
	```

Notes and recommendations
- `ifBO` README recommends Python 3.11; verify your `uv` environment uses
	a compatible Python version before installing.
- We pinned the current commit of `icml-2024` used for this workspace to the
	commit `421ff96986c8efbe92bd2ab64a4bab49b0dd46fb` in
	`requirements-ifbo.txt` to ensure reproducible installs.

