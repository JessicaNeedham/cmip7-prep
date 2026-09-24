""" 
Utility functions for processing multi-plexed FATES dimensions prior to cmorization

Usage:
    python deduplex_fates_dims.py --inputdir <inputdir> --outputdir <outputdir> --varlist <varlist> 
"""

import numpy as np
import xarray as xr

import argparse
import os
from pathlib import Path

def scpf_to_scls_by_pft(scpf_var, dataset):
    """function to reshape a fates multiplexed size and pft-indexed variable to one indexed by size class and pft
    first argument should be an xarray DataArray that has the FATES SCPF dimension or a string with the name of the variable in the dataset,
    second argument should be an xarray Dataset that has the FATES SCLS dimension
    (possibly the dataset encompassing the dataarray being transformed)
    returns the dataset with the variable by both dimension"""
    return deduplex(dataset, scpf_var, "scls", "pft")

def deduplex(dataset, this_var, dim1_short, dim2_short):
    """Reshape a duplexed FATES dimension into its constituent dimensions

    For example, given a variable with dimensions
        (time, fates_levagepft, lat, lon),
    this will return a Dataset with new variables one with dimensions
        (time, fates_levage, lat, lon) and one with (time, fates_levpft, lat, lon).

    Args:
        dataset (xarray Dataset): Dataset containing the variable with dimension to de-duplex
        this_var (string or xarray DataArray): (Name of) variable with dimension to de-duplex
        dim1_short (string): Short name of first duplexed dimension. E.g., when de-duplexing
                             fates_levagepft, dim1_short=age.
        dim2_short (string): Short name of second duplexed dimension. E.g., when de-duplexing
                             fates_levagepft, dim2_short=pft.
        
    Raises:
        RuntimeError: dim1_short == dim2_short (not yet handled)
        TypeError: Incorrect type of this_var
        NameError: Dimension not found on Dataset

    Returns:
        xarray Dataset
    """

    if dim1_short == dim2_short:
        raise RuntimeError("deduplex() can't currently handle dim1_short==dim2_short")

    # Get DataArray
    if isinstance(this_var, xr.DataArray):
        da_in = this_var
    elif isinstance(this_var, str):
        da_in = dataset[this_var]
    else:
        raise TypeError("this_var must be either string or DataArray, not " + type(this_var))

    # Get combined dim name
    dim_combined = _get_dim_combined(dim1_short, dim2_short)
    if dim_combined not in da_in.dims:
        raise NameError(f"Dimension {dim_combined} not present in DataArray with dims {da_in.dims}")

    # Get individual dim names
    dim1 = _get_check_dim(dim1_short, dataset)
    dim2 = _get_check_dim(dim2_short, dataset)

    # Split multiplexed dimension into its components
    n_dim1 = len(dataset[dim1])
    da_out = (
        da_in.rolling({dim_combined: n_dim1}, center=False)
        .construct(dim1)
        .isel({dim_combined: slice(n_dim1 - 1, None, n_dim1)})
        .rename({dim_combined: dim2})
        .assign_coords({dim1: dataset[dim1]})
        .assign_coords({dim2: dataset[dim2]})
    )

    da_dim1 = da_out.sum(dim=dim2)
    da_dim2 = da_out.sum(dim=dim1)

    dim1_short_name = _get_short_dim_name(dim1_short)
    dim2_short_name = _get_short_dim_name(dim2_short)

    # get the first part of the var name before the duplexed dimension and append the new dim names   
    new_var_name1 = this_var.rsplit("_", 1)[0] + "_" + dim1_short_name
    new_var_name2 = this_var.rsplit("_", 1)[0] + "_" + dim2_short_name

    dataset[new_var_name1] = da_dim1
    dataset[new_var_name2] = da_dim2

    return dataset

def _get_dim_combined(dim1_short, dim2_short):
    """Get duplexed dimension name, given two short names

    Args:
        dim1_short (string): Short name of first duplexed dimension. E.g., when de-duplexing
                             fates_levscpf, dim1_short=scls.
        dim2_short (string): Short name of second duplexed dimension. E.g., when de-duplexing
                             fates_levscpf, dim2_short=pft.

    Returns:
        string: Duplexed dimension name
    """
    dim_combined = "fates_lev" + dim1_short + dim2_short

    # Handle further-shortened dim names
    if dim_combined == "fates_levcanleaf":
        dim_combined = "fates_levcnlf"
    elif dim_combined == "fates_levcanpft":
        dim_combined = "fates_levcapf"
    elif dim_combined == "fates_levcdamscls":
        dim_combined = "fates_levcdsc"
    elif dim_combined == "fates_levsclsage":
        dim_combined = "fates_levscag"
    elif dim_combined == "fates_levsclspft":
        dim_combined = "fates_levscpf"
    elif dim_combined == "fates_levlandusepft":
        dim_combined = "fates_levlupft"
    elif dim_combined == "fates_levcanleaf":
        dim_combined = "fates_levcnlf"

    return dim_combined

def _get_check_dim(dim_short, dataset):
    """Get dim name from short code and ensure it's on Dataset

    Probably only useful internally to this module; see deduplex().

    Args:
        dim_short (string): The short name of the dimension. E.g., "age"
        dataset (xarray Dataset): The Dataset we expect to include the dimension

    Raises:
        NameError: Dimension not found on Dataset

    Returns:
        string: The long name of the dimension. E.g., "fates_levage"
    """

    dim = "fates_lev" + dim_short
    if dim not in dataset.dims:
        raise NameError(f"Dimension {dim} not present in Dataset with dims {dataset.dims}")
    return dim


def _get_short_dim_name(dim_short):
    if dim_short == 'scls':
        dim_name_short = "SZ"
    elif dim_short == 'pft':
        dim_name_short = "PF"
    return dim_name_short        


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Utility functions for processing multi-plexed FATES dimensions prior to cmorization"
    )
    parser.add_argument(
        "--inputdir",
        type=str,
        required=True,
        help="Directory containing input files",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        required=True,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--varlist",
        type=str,
        required=True,
        help="Comma-separated list of variables to process",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    inputdir = Path(args.inputdir)
    outputdir = Path(args.outputdir)
    varlist = [var.strip() for var in args.varlist.split(",")]

    # Ensure output directory exists
    os.makedirs(outputdir, exist_ok=True)
    
    # Process each file in the input directory
    for file_path in inputdir.glob("*.nc"):
        
        dataset = xr.open_dataset(file_path)
    
        # Process each variable in the varlist
        for var in varlist:
            if var in dataset:
                deduplexed_dataset = scpf_to_scls_by_pft(var, dataset)
               
        # Save the modified dataset to the output directory
        output_file_path = outputdir / file_path.name
        deduplexed_dataset.to_netcdf(output_file_path)
        print(f"Processed {file_path} and saved to {output_file_path}")

