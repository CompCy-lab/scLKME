from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.random import RandomState
from scanpy.tools._utils import _choose_representation
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array, check_random_state

from ._utils import random_feats

_KernelCapable = Literal[
    "additive_chi2",
    "chi2",
    "linear",
    "poly",
    "polynomial",
    "rbf",
    "laplacian",
    "sigmoid",
    "cosine",
]

SHIFT_INVARIANT_KERNEL = ["rbf"]


def kernel_mean_embedding(
    adata: AnnData,
    partition_key: str,
    X_anchor: np.ndarray,
    n_pcs: Optional[str] = None,
    use_rep: Optional[str] = None,
    random_state: Union[None, int, RandomState] = 0,
    method: Literal["exact", "approx"] = "exact",
    kernel: _KernelCapable = "rbf",
    kernel_kwds: Dict[str, Any] = {},
    n_jobs: Optional[int] = None,
    key_added: Optional[str] = None,
    copy=False,
) -> Optional[AnnData]:
    """\
    Compute the kernel mean embedding

    """
    # first we run the sketching algorithm to find the anchors
    adata = adata.copy() if copy else adata

    if key_added is None:
        key_added = "kme"
    else:
        key_added = key_added + "_kme"

    adata.uns[key_added] = {}

    kme_dict = adata.uns[key_added]

    kme_dict["partition_key"] = partition_key
    kme_dict["params"] = {"kernel": kernel, "method": method}
    kme_dict["params"]["random_state"] = random_state
    if kernel_kwds:
        kme_dict["params"]["kernel_kwds"] = kernel_kwds

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, silent=True)
    if issparse(X):
        X = X.toarray()
    if use_rep is not None:
        kme_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        kme_dict["params"]["n_pcs"] = n_pcs

    # check the input anchor matrix
    X_anchor = check_array(
        X_anchor, dtype=np.float32, ensure_2d=True, accept_sparse=False
    )
    if X_anchor.shape[1] != X.shape[1]:
        raise ValueError(
            f"the shape: {X_anchor.shape} of input anchor matrix \
            cannot match the data representation shape in `adata`."
        )

    if method == "exact":
        from sklearn.metrics import pairwise_kernels

        # compute the kernels between X and X_anchor
        if kernel == "rbf":
            X_kme = rbf_kernel(
                X, X_anchor, gamma=kernel_kwds.get("gamma", 1), n_jobs=n_jobs
            )
        elif kernel == "laplacian":
            X_kme = laplacian_kernel(
                X=X, Y=X_anchor, gamma=kernel_kwds.get("gamma", 1), n_jobs=n_jobs
            )
        else:
            X_kme = pairwise_kernels(
                X=X, Y=X_anchor, metric=kernel, n_jobs=n_jobs, **kernel_kwds
            )

        # make the kernel matrix sparse if necessary
        mask = X_kme > 1e-14
        X_kme[~mask] = 0

    elif method == "approx":
        if kernel not in SHIFT_INVARIANT_KERNEL:
            raise ValueError(
                f"Only shift invariant kernels: [{','.join(SHIFT_INVARIANT_KERNEL)}] "
                "support `method=approx` for kernel mean embedding."
            )
        kme_dict["params"]["kernel_kwds"]["gamma"] = kernel_kwds.get("gamma", 1)
        kme_dict["params"]["kernel_kwds"]["D"] = kernel_kwds.get("D", 2000)

        random_state = check_random_state(random_state)
        phi_X = random_feats(
            X,
            gamma=kme_dict["params"]["kernel_kwds"]["gamma"],
            D=kme_dict["params"]["kernel_kwds"]["D"],
            random_state=random_state,
        )
        phi_X_anchor = random_feats(
            X_anchor,
            gamma=kme_dict["params"]["kernel_kwds"]["gamma"],
            D=kme_dict["params"]["kernel_kwds"]["D"],
            random_state=random_state,
        )
        # outer multiplication
        X_kme = phi_X @ phi_X_anchor.T

    adata.obsm["X_kme"] = X_kme  # save the results

    # save the resulting embeddings based on partition_key
    df_partition = pd.DataFrame(
        adata.obsm["X_kme"].copy(),
        columns=[f"X_kme_{i}" for i in range(adata.obsm["X_kme"].shape[1])],
    )
    df_partition[partition_key] = adata.obs[partition_key].tolist()
    df_partition = df_partition.groupby(partition_key, sort=False).aggregate(np.mean)
    kme_dict[f"{partition_key}_kme"] = df_partition

    return adata if copy else None


def rbf_kernel(X, Y=None, gamma=None, normalized=True, n_jobs=None):
    """Compute the rbf (gaussian) kernel between X and Y

        K(X, Y) = exp(-gamma ||x - y||^2 / Den)

    for each pair of rows in X and y in Y.
    Note: Den := np.max(Dsq, axis=0) / 4

    """
    Dsq = pairwise_distances(X, Y, metric="euclidean", squared=True, n_jobs=n_jobs)
    Den = np.max(Dsq, axis=0) / 4 if normalized else 1

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.exp(-gamma * Dsq / Den)
    return K


def laplacian_kernel(X, Y, gamma=None, normalized=True, n_jobs=None):
    """Compute the laplacian kernel between X and Y.

        K(X, Y) = exp(-gamma ||x - y||_1 / Den)

    for each pair of rows x in X and y in Y.
    """
    D_manhattan = pairwise_distances(X, Y, metric="manhattan", n_jobs=n_jobs)
    Den = np.max(D_manhattan, axis=0) / 4 if normalized else 1

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.exp(-gamma * D_manhattan / Den)
    return K
