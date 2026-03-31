


## Environment Setup

We use the `uv` package manager


Install the uv package manager through the command line:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
From the repository root run:
```
uv sync
```

Code can be ran via the `uv run <script>` commands and a jupyter notebook can be loaded via `uv run --with jupyter jupyter lab`.
```
uv run --with ipython --with ipykernel ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=climate-data-hackathon
```

```
uv run --with jupyter jupyter lab
```

Files to be opened in the jupyter notebook are:
+ `/tests/dynamic_co2_with_package.ipynb`
+ `/Weronikas Code/Forecasting_co2.ipynb`
+ `Weronikas Code/co2_emission_plot_with_T_anom.ipynb`

