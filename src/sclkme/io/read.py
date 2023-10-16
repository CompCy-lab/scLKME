import functools
import gc
import os
from typing import Any, Callable, Dict, List, Optional, Union

import anndata
import pandas as pd
import readfcs
from anndata import AnnData
from geosketch import gs
from numpy import random
from PyCytoData import PyCytoData
from rich.console import Console
from scanpy._utils import sanitize_anndata

from sclkme.utils import MpRunner

logger = Console(log_path=False)


def prepare_cytometry_data(
    data_dir: str,
    metadata: str,
    filename_col: str,
    lineage_channels: List,
    groupby: Optional[str] = None,
    down_sample: Optional[int] = None,
    num_workers: int = -1,
    random_state: Union[None, int, random.RandomState] = 0,
    gate_kwds: Dict[str, Any] = {},
    verbose: bool = False,
) -> Union[AnnData, Dict[str, AnnData]]:
    # read the metadata first
    meta_df = pd.read_csv(metadata, header=0)

    # check if key exists
    assert (filename_col in meta_df.columns) and (
        groupby in meta_df.columns if groupby is not None else True
    ), f"the meta data file: {metadata} doesn't have the filename column: {filename_col}"

    task_handler = functools.partial(
        read_data,
        data_dir=data_dir,
        filename_col=filename_col,
        lineage_channels=lineage_channels,
        down_sample=down_sample,
        replace=False,
        random_state=random_state,
        gate_kwds=gate_kwds,
        verbose=verbose,
    )
    logger.log(
        f"Reading and preprocessing {len(meta_df[filename_col])} files. It may take a few minutes..."
    )

    adata = None
    if groupby is not None:
        adata = dict()
        for grp_name, grp in meta_df.groupby(groupby, sort=False):
            adata[grp_name] = _read_handler(
                grp, func=task_handler, num_workers=num_workers
            )
    else:
        adata = _read_handler(meta_df, func=task_handler, num_workers=num_workers)
    logger.log("Your data is converted into `AnnData`.")

    if adata is None:
        raise RuntimeError(
            "Reading data failed, please check the input and try again..."
        )

    logger.log("Done!")

    return adata


# read the data
def _read_handler(rows_df: pd.DataFrame, func: Callable, num_workers: int):
    # start multiprocessing pool to handle jobs
    multiprocessing_runner = MpRunner(num_workers)
    results = multiprocessing_runner.run(
        [rows_df.iloc[i, :] for i in range(len(rows_df.index))], func
    )

    assert len(rows_df.index) == len(results), "input job length != result length"

    adata_list = [None for _ in range(len(rows_df.index))]
    for res in results:
        if res.run_ok:
            if res.result["output"] is None:
                raise RuntimeError("resulting adata is None, exit...")
            adata_list[res.jobid] = res.result["output"]
        else:
            raise RuntimeError(
                f"Error occurred in reading the processing {res.result['msg']}"
            )

    for jobid, adata in enumerate(adata_list):
        if adata is None:
            raise RuntimeError(f"job: {jobid} is not correctly handled.")

    # merge the data
    adata = anndata.concat(adata_list, merge="same", index_unique="-", axis=0)  # type: ignore
    adata.obs.reset_index(drop=True, inplace=True)
    sanitize_anndata(adata)

    del adata_list, results
    gc.collect()

    return adata


def read_data(
    row: pd.Series,
    data_dir: str,
    filename_col: str,
    lineage_channels: str,
    down_sample: Optional[int] = None,
    replace: bool = False,
    random_state: Union[None, int, random.RandomState] = 0,
    gate_kwds: Dict[str, Any] = None,
    verbose: bool = False,
):
    if verbose:
        logger.log(f"Reading and preprocessing file: {row[filename_col]}")

    if row[filename_col].endswith(".fcs"):
        adata = readfcs.read(os.path.join(data_dir, row[filename_col]), reindex=False)
        assert (
            "channel" in adata.var.columns
        ), f"the col: channel cannot be found in the fcs file: {row[filename_col]}"
        adata.var.set_index("channel", inplace=True)
    elif row[filename_col].endswith(".csv"):
        adata = anndata.read_csv(os.path.join(data_dir, row[filename_col]))
    elif row[filename_col].endswith(".tsv"):
        adata = anndata.read_text(
            os.path.join(data_dir, row[filename_col]), delimiter="\t"
        )
    else:
        raise ValueError(
            "the file extension is not understood for reading, please use .fcs, .csv and .tsv file."
        )

    channels = adata.var.index.tolist()
    expr = PyCytoData(
        expression_matrix=adata.X.copy(),
        channels=channels,
        lineage_channels=lineage_channels,
    )
    if gate_kwds is not None:
        expr.preprocess(**gate_kwds, verbose=verbose)
    adata = anndata.AnnData(X=expr.expression_matrix, obs=row.to_dict(), var=adata.var)

    if down_sample is not None and adata.n_obs >= down_sample:
        if verbose:
            logger.log(f"downsample to {down_sample} cells.")
        gs_idx = gs(
            adata[:, lineage_channels].X,
            down_sample,
            replace=replace,
            seed=random_state,
        )
        adata = adata[gs_idx].copy()

    return adata
