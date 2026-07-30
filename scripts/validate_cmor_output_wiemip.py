#!/usr/bin/env python3

"""Quick-check validation for CMORized CMIP7 output trees.

Specifically for WIEMIP this script produces timeseries plots for all 
variables (not just the ones expected by cmip)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from cmip7_prep.mapping_compat import packaged_mapping_resource

from cmor_driver import REALM_YAML_MAP

logger = logging.getLogger("cmip7_prep.validate_cmor_output")

MODEL_NAMING_MAPS = {
    "noresm": ["NCC", "NorESM3"],
    "cesm": ["NCAR", "CESM3"],
}

CANONICAL_REALM_MAP = {
    "aerosol": "atmos",
    "atmosChem": "atmos",
    "ocnBgchem": "ocean",
}

LOG_NAME_RE = re.compile(r"^cmor_\d{8}T\d{6}Z_(?P<variable>.+)\.log$")
LOG_ERROR_RE = re.compile(r"Error: ", re.IGNORECASE)
LOG_SUCCESS_RE = re.compile(r"\b(success|complete(?:d)?|finished)\b", re.IGNORECASE)


@dataclass
class LogRecord:
    """A parsed record for one CMOR log file."""

    variable: str
    path: str
    has_error: bool
    has_success_marker: bool
    error_lines: list[str]


@dataclass
class ProducedFileRecord:
    """Inventory record for one produced CMOR file."""

    variable: str
    consortium: str
    model: str
    experiment: str
    ensemble_member: str
    region: str
    frequency: str
    dimension_folder: str
    grid_type: str
    file_path: str
    data_var: str
    dims: list[str]
    sizes: dict[str, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick-check validation for CMIP7 CMOR output trees"
    )
    parser.add_argument(
        "--model",
        choices=sorted(REALM_YAML_MAP),
        required=True,
        help="Model whose CMIP7 output should be validated",
    )
    parser.add_argument(
        "--realm",
        choices=[
            "atmos",
            "aerosol",
            "atmosChem",
            "land",
            "landIce",
            "ocean",
            "ocnBgchem",
            "seaIce",
        ],
        required=True,
        help="Realm to validate; aerosol/atmosChem map to atmos YAML, ocnBgchem maps to ocean YAML",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="piControl",
        help="Experiment name to validate (Default piControl)",
    )
    parser.add_argument(
        "--frequency",
        choices=["mon", "day", "6hr", "3hr", "yr", "fx"],
        required=True,
        help="CMIP7 output frequency to validate",
    )
    parser.add_argument(
        "--root-output-path",
        required=True,
        help="Root output directory containing CMIP7/ and logs/",
    )
    parser.add_argument(
        "--ensemble-member",
        default=None,
        help="Optional ensemble member filter",
    )
    parser.add_argument(
        "--custom-yaml",
        default=None,
        help="Optional custom YAML mapping file",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional explicit list of branded variable names to validate",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for validation reports; defaults to <root-output-path>/validation_reports/<subset>",
    )
    parser.add_argument(
        "--plot-timeseries",
        action="store_true",
        help="Create composite mean time-series plots for produced variables",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Create per-variable time-mean maps where possible",
    )
    parser.add_argument(
        "--plot_dir",
        default=None,
        help="Output directory for plot files; defaults to <report-dir>/plots",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=36,
        help="Maximum number of variables to plot per plot mode",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if missing variables or log errors are found",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def canonical_realm(realm: str) -> str:
    """Return the YAML realm corresponding to the requested realm."""
    return CANONICAL_REALM_MAP.get(realm, realm)


def resolve_cmip_root(root_output_path: str | Path) -> Path:
    """Resolve the CMIP7 root directory from either a parent or direct path."""
    root = Path(root_output_path).expanduser().resolve()
    if (root / "CMIP").is_dir():
        return root
    cmip_root = root / "CMIP7"
    if (cmip_root / "CMIP").is_dir():
        return cmip_root
    raise FileNotFoundError(
        f"Could not locate CMIP output under {root}. Expected either {root / 'CMIP7' / 'CMIP'} or {root / 'CMIP'}."
    )


def resolve_logs_dir(root_output_path: str | Path, cmip_root: Path) -> Path:
    """Resolve the logs directory adjacent to the CMIP7 output tree."""
    root = Path(root_output_path).expanduser().resolve()
    if (root / "logs").is_dir():
        return root / "logs"
    if (cmip_root.parent / "logs").is_dir():
        return cmip_root.parent / "logs"
    return root / "logs"


def get_yaml_path(model: str, realm: str, custom_yaml: str | None) -> Path:
    """Resolve the YAML mapping path for the selected model/realm."""
    if custom_yaml:
        path = Path(custom_yaml).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    yaml_realm = canonical_realm(realm)
    yaml_name = REALM_YAML_MAP.get(model, {}).get(yaml_realm)
    if yaml_name is None:
        raise ValueError(f"No YAML mapping defined for model={model}, realm={realm}")
    with packaged_mapping_resource(yaml_name) as resource_path:
        return Path(resource_path)


def load_yaml_variables(yaml_path: Path) -> dict[str, dict[str, Any]]:
    """Load the variables block from a CMIP7 mapping YAML file."""
    with open(yaml_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    variables = payload.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(
            f"Unsupported YAML structure in {yaml_path}: expected top-level variables dict"
        )
    return variables

def open_dataset_inventory(
    file_path: Path, variable: str
) -> tuple[str, list[str], dict[str, int]]:
    """Open a CMOR file and inspect the target data variable dimensions."""
    with xr.open_dataset(file_path, decode_times=False) as dataset:
        if variable in dataset.data_vars:
            data_var = variable
        else:
            candidates = [
                name
                for name in dataset.data_vars
                if not name.endswith("_bnds")
                and name not in {"time_bnds", "lat_bnds", "lon_bnds"}
            ]
            if not candidates:
                raise ValueError(f"No data variables found in {file_path}")
            data_var = candidates[0]
        array = dataset[data_var]
        dims = list(array.dims)
        sizes = {dim: int(array.sizes[dim]) for dim in array.dims}
    return data_var, dims, sizes


def scan_output_tree(
    cmip_root: Path,
    *,
    model: str,
    experiment: str,
    frequency: str,
    ensemble_member: str | None = None,
) -> tuple[dict[str, list[Path]], list[ProducedFileRecord], list[dict[str, str]]]:
    """Scan the CMIP7 tree and inventory produced files for the selected subset."""
    produced: dict[str, list[Path]] = defaultdict(list)
    inventory: list[ProducedFileRecord] = []
    inspection_errors: list[dict[str, str]] = []
    institution_id = MODEL_NAMING_MAPS[model][0]
    pattern = cmip_root.glob(
        f"CMIP/{institution_id}/{MODEL_NAMING_MAPS[model][1]}/{experiment}/*/glb/{frequency}/*/*/*/*.nc"
    )
    for file_path in sorted(pattern):
        relative = file_path.relative_to(cmip_root)
        # sys.exit(4)
        parts = relative.parts
        if len(parts) < 11:
            logger.debug("Skipping unexpected output path layout: %s", file_path)
            continue

        (
            _,
            consortium,
            path_model,
            path_experiment,
            path_ensemble,
            region,
            path_frequency,
            variable,
            dimension_folder,
            grid_type,
            _,
        ) = parts[:11]

        if institution_id and consortium != institution_id:
            logger.debug("Skipping due to institution_id filter: %s", institution_id)
            continue
        if ensemble_member and path_ensemble != ensemble_member:
            logger.debug("Skipping due to ensemble filter: %s", ensemble_member)
            continue
        produced[f"{variable}_{dimension_folder}"].append(file_path)
        try:
            data_var, dims, sizes = open_dataset_inventory(file_path, variable)
        except Exception as exc:  # pylint: disable=broad-except
            inspection_errors.append(
                {
                    "variable": variable,
                    "path": str(file_path),
                    "error": repr(exc),
                }
            )
            continue

        inventory.append(
            ProducedFileRecord(
                variable=variable,
                consortium=consortium,
                model=path_model,
                experiment=path_experiment,
                ensemble_member=path_ensemble,
                region=region,
                frequency=path_frequency,
                dimension_folder=dimension_folder,
                grid_type=grid_type,
                file_path=str(file_path),
                data_var=data_var,
                dims=dims,
                sizes=sizes,
            )
        )
    return produced, inventory, inspection_errors


def _open_variable_timeseries(
    file_paths: list[Path], variable: str
) -> xr.DataArray | None:
    """Open a variable across files and reduce it to a 1D time series if possible."""
    if not file_paths:
        return None
    with xr.open_mfdataset(
        file_paths, combine="by_coords", decode_times=True
    ) as dataset:
        data_var = (
            variable if variable in dataset.data_vars else list(dataset.data_vars)[0]
        )
        array = dataset[data_var]
        if "time" not in array.dims:
            return None
        reduce_dims = [
            dim for dim in array.dims if dim != "time" and not dim.endswith("bnds")
        ]
        if reduce_dims:
            array = array.mean(dim=reduce_dims, skipna=True)
        return array.load()


def _open_variable_map(file_paths: list[Path], variable: str) -> xr.DataArray | None:
    """Open a variable across files and reduce it to a 2D spatial field if possible."""
    if not file_paths:
        return None
    with xr.open_mfdataset(
        file_paths, combine="by_coords", decode_times=True
    ) as dataset:
        data_var = (
            variable if variable in dataset.data_vars else list(dataset.data_vars)[0]
        )
        array = dataset[data_var]
        if "time" in array.dims:
            array = array.mean(dim="time", skipna=True)
        spatial_dims = [
            dim
            for dim in array.dims
            if dim.lower()
            in {"lat", "lon", "latitude", "longitude", "xh", "yh", "i", "j"}
        ]
        if len(spatial_dims) != 2:
            return None
        extra_dims = [dim for dim in array.dims if dim not in spatial_dims]
        if extra_dims:
            array = array.isel({dim: 0 for dim in extra_dims})
        return array.load()


def create_timeseries_plots(
    produced_files: dict[str, list[Path]],
    plot_dir: Path,
    max_plots: int,
) -> list[str]:
    """Create paginated composite mean time-series plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping time-series plots")
        return []

    plotted = []
    variables = sorted(produced_files)[:max_plots]
    if not variables:
        return plotted

    page_size = 9
    for page_index in range(0, len(variables), page_size):
        page_variables = variables[page_index : page_index + page_size]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
        for axis, variable in zip(axes.flat, page_variables):
            series = _open_variable_timeseries(
                produced_files[variable], variable.split("_")[0]
            )
            if series is None:
                axis.set_title(variable)
                axis.text(
                    0.5, 0.5, "No plottable time series", ha="center", va="center"
                )
                axis.set_axis_off()
                continue
            axis.plot(get_plottble_times(series), series.values, linewidth=1.0)
            axis.set_title(variable)
            axis.tick_params(axis="x", rotation=30)
            axis.set_xlabel("Time (years)")
            axis.set_ylabel(
                f"{variable.split('_')[0]} ({series.attrs.get('units', 'unknown')})"
            )
        for axis in axes.flat[len(page_variables) :]:
            axis.set_axis_off()

        fig.tight_layout()
        output_path = (
            plot_dir / f"timeseries_composite_{page_index // page_size + 1:02d}.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted.append(str(output_path))
    return plotted


def get_plottble_times(tseries: xr.DataArray) -> np.ndarray:
    monlength = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    strings = [date.strftime("%Y-%m-%d") for date in tseries["time"].values]
    numbers_lists = [strings.split("-") for strings in strings]
    numbers = np.array(
        [
            (365 * int(val[0]) + monlength[: int(val[1])].sum() + int(val[2])) / 365.0
            for val in numbers_lists
        ],
        dtype=float,
    )
    return numbers


def create_map_plots(
    produced_files: dict[str, list[Path]],
    plot_dir: Path,
    max_plots: int,
) -> list[str]:
    """Create per-variable time-mean map plots where the data shape allows it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping map plots")
        return []

    plotted = []
    for variable in sorted(produced_files)[:max_plots]:
        field = _open_variable_map(produced_files[variable], variable.split("_")[0])
        if field is None:
            continue
        fig, axis = plt.subplots(figsize=(8, 4.5))
        field.plot(ax=axis)
        axis.set_title(f"{variable} time-mean")
        output_path = plot_dir / f"map_{variable}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plotted.append(str(output_path))
    return plotted





def main() -> int:
    """Run the CMOR output validation workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cmip_root = resolve_cmip_root(args.root_output_path)
    logs_dir = resolve_logs_dir(args.root_output_path, cmip_root)

    yaml_path = get_yaml_path(args.model, args.realm, args.custom_yaml)
    yaml_variables = load_yaml_variables(yaml_path)

    produced_files, inventory_records, inspection_errors = scan_output_tree(
        cmip_root,
        model=args.model,
        experiment=args.experiment,
        frequency=args.frequency,
        ensemble_member=args.ensemble_member,
    )

    plot_outputs = {"timeseries": [], "maps": []}
    if args.plot_timeseries or args.plot_maps:
        plot_dir = (
            Path(args.plot_dir).expanduser().resolve()
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        if args.plot_timeseries:
            plot_outputs["timeseries"] = create_timeseries_plots(
                produced_files, plot_dir, args.max_plots
            )
        if args.plot_maps:
            plot_outputs["maps"] = create_map_plots(
                produced_files, plot_dir, args.max_plots
            )

    if args.strict and (
        report["variables_with_log_errors"] or report["expected_but_not_produced"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
