import argparse

import os
import sys
import glob
import logging
import shutil


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("cmip7_prep.rename_files_for_wiemip")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Renaming script to rename CMIP7 output files to match Wiemip naming conventions for a given experiment"
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
        default="land",
        help="Realm to rename files for",
    )
    parser.add_argument(
        "--experiment-original",
        type=str,
        default="piControl",
        help="Experiment name to rename from",
    )
    parser.add_argument(
        "--experiment-rename",
        type=str,
        required=True,
        help="Experiment name to rename to",
    )
    parser.add_argument(
        "--frequency",
        choices=["mon", "day", "6hr", "3hr", "yr", "fx", "all"],
        default="all",
        help="CMIP7 output frequency to validate",
    )
    parser.add_argument(
        "--resolution",
        default="1",
        help="Resolution specification string, assuming 1 by 1 degree with designation 1",
    )
    parser.add_argument(
        "--root-output-path",
        required=True,
        help="Root output directory containing CMIP7/ and logs/ from cmor_driver.py output",
    )
    parser.add_argument(
        "--custom-yaml",
        default=None,
        help="Optional custom YAML mapping file",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default="*",
        help="Optional explicit list of branded variable names to validate",
    )
    parser.add_argument(
        "--clim-force-model",
        default="ukesm",
        help="Climate model provided for forcing",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plot files; defaults to <report-dir>/plots",
    )
    parser.add_argument(
        "--delete-originals",
        type=int,
        default=False,
        help="Whether to delete original files after renaming (default: False)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if missing variables or log errors are found",
    )
    parser.add_argument(
        "--factorial",
        default=None,
        help="Optional factorial e.g. noFire",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.setLevel(args.log_level)

    print(args)
    if not os.path.isdir(args.root_output_path):
        logger.error(f"Root output path {args.root_output_path} does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.isdir(f"{args.root_output_path}/CMIP7"):
        logger.error(f"CMIP7 directory not found in root output path {args.root_output_path}.")
        sys.exit(1)
    
    # Construct the glob pattern for the files to rename
    if args.frequency == "all":
        frequency_pattern = "*"
    else:
        frequency_pattern = args.frequency

    if args.output_dir is None:
        args.output_dir = os.path.join(args.root_output_path, "WIEMIP_renamed_files")
    os.makedirs(args.output_dir, exist_ok=True)
    glob_pattern = f"{args.root_output_path}/CMIP7/CMIP/NCC/NorESM3/{args.experiment_original}/*/glb/{frequency_pattern}/{args.variables}/*/*/*.nc"
    files_to_rename = glob.glob(glob_pattern)
    for file_path in files_to_rename:
        parts = file_path.split("/")
        cmip7_compound_name = f"land.{parts[-4]}.{parts[-3]}.{parts[-5]}.{parts[-6]}"
        if args.factorial is None:
            new_file_name = f"FATES_{args.clim_force_model}_{args.experiment_rename}_{cmip7_compound_name}_{args.resolution}.nc"
        else:
            new_file_name = f"FATES_{args.clim_force_model}_{args.experiment_rename}_{cmip7_compound_name}_{args.factorial}_{args.resolution}.nc"
        
        print(f"{file_path} -> {args.output_dir}/{new_file_name}")

        new_file_path = os.path.join(args.output_dir, new_file_name)
        shutil.copy(file_path, new_file_path)
        if args.delete_originals:
            os.remove(file_path)

if __name__ == "__main__":
    main()
