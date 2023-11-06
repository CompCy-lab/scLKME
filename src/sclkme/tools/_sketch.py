import warnings
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import scipy
from anndata import AnnData
from scanpy._utils import AnyRandom
from scanpy.tools._utils import _choose_representation
from sklearn.utils import check_random_state

from ._utils import random_feats


def sketch(
    adata: AnnData,
    n_sketch: int,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    random_state: AnyRandom = 0,
    method: Literal["gs", "kernel_herding", "random"] = "gs",
    replace: bool = False,
    inplace: bool = False,
    sketch_kwds: Dict[str, Any] = {},
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Cell sketching :cite:`Hie19gs` :cite:`Vishal22`

    *sketch* is a function that can be used for summarizing the landscape of
    a large single-cell dataset by selecting a compact subset of cells. This
    function supports three different methods to perform cell sketching:
    geoemetric sketching :cite:`Hie19gs`, kernel herding :cite:`Vishal22`, and simple
    random sampling without replacement.

    .. note::
       More information and bug reports about the geometric sketching method `here
       <https://github.com/brianhie/geosketch>`__.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_sketch
        number of subsampled cells
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically:
        For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
        If 'X_pca' is not present, it's computed with default parameters.
    n_pcs
        Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`
    random_state
        A numpy random seed.
    method
        Use 'gs' :cite:`Hie19gs`, 'kernel_herding' :cite:`Vishal22` or 'random' for
        for summarizing the landscape of large single-cell datasets.
    replace
        whether to sample with replacement (default=False)
    inplace
        It is a boolean which subsample cells in `adata` itself if True.
    sketch_kwds
        A dict that holds other specific parameters used by the sketching methods.
        (e.g. scaling parameter `gamma` and dimensionality of the random Fourier
        frequency features `D` in the kernel herding method.)
    key_added
        If not specified, the subsampled index in boolean format is stored in
        .obs['sketch'], and the sketching parameters are stored in .uns['sketch'].
        If specified, the subsampled index in boolean format is stored in .obs[key_added+'sketch'],
        and the sketching parameters are stored in .uns[key_added+'sketch'].
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of subsampled indices.

    Example
    -------
    >>> import scanpy as sc
    >>> import sclkme

    Load annotated dataset:

    >>> adata = sc.datasets.pbmc3k_processed()

    Run the cell sketching using geometric sketching:

    >>> sclkme.tl.sketch(adata, n_sketch=128, use_rep="X_pca", method="gs", key_added = "gs")

    Run the cell sketching using kernel herding:

    >>> sclkme.tl.sketch(adata, n_sketch=128, use_rep="X_pca", method="kernel_herding", key_added = "kh")

    Visualize the sketched dataset:

    >>> adata_sketch_gs = adata[adata.obs['gs_sketch']]
    >>> sc.pl.umap(adata_sketch_gs, color="louvain", size=100)
    >>> adata_sketch_kh = adata[adata.obs['kh_sketch']]
    >>> sc.pl.umap(adata_sketch_kh, color="louvain", size=100)
    """
    # start = logg.info("Cell sketching")

    if method == "gs":
        try:
            from geosketch import gs
        except ImportError:
            raise ImportError(
                "Please install the package of geometric sketching:\n\t `pip install geosketch`."
            )

    adata = adata.copy() if copy else adata
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if key_added is None:
        key_added = "sketch"
    else:
        key_added = key_added + "_sketch"

    adata.uns[key_added] = {"params": {"n_sketch": n_sketch}, "method": method}
    if use_rep is not None:
        adata.uns[key_added]["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        adata.uns[key_added]["params"]["n_pcs"] = n_pcs

    adata.uns[key_added]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    if n_sketch >= adata.n_obs:
        replace = True
        warnings.warn("n_sketch too large, adjusting to sampling with replacement.")

    # find the representation for sketching
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, silent=True)
    # main routine
    if method == "gs":
        # get the seed from random state
        random_seed = random_state.get_state()[1][0]
        sketch_index = gs(X, n_sketch, seed=random_seed, replace=replace)

    elif method == "kernel_herding":
        if replace:
            raise ValueError(
                "`kernel herding` cannot perform sampling with replacement, "
                "set `rsampling=False` or decrease the size of `n_sketch`."
            )
        if isinstance(X, scipy.sparse.csr_matrix):
            X = X.toarray()

        if sketch_kwds:
            for key in sketch_kwds:
                adata.uns[key_added]["params"][key] = sketch_kwds[key]

        adata.uns[key_added]["params"]["gamma"] = sketch_kwds.get("gamma", 1)
        adata.uns[key_added]["params"]["D"] = sketch_kwds.get("D", 2000)
        phi_X = random_feats(
            X,
            gamma=adata.uns[key_added]["params"]["gamma"],
            D=adata.uns[key_added]["params"]["D"],
            random_state=random_state,
        )
        sketch_index, _ = kernel_herding(X, phi_X, n_sketch)

    elif method == "random":
        size = adata.n_obs
        sketch_index = random_state.choice(
            np.arange(size), size=n_sketch, replace=replace
        )
    else:
        raise ValueError(
            f"method: {method} is not supported, use `gs`, `kernel_herding` or `random` instead."
        )

    # sort the indices
    sketch_index = np.sort(sketch_index)
    if inplace:
        adata = adata[sketch_index, :].copy()
    else:
        adata.obs[key_added] = False
        adata.obs.loc[adata.obs.index[sketch_index], key_added] = True

    # logg.info(
    #     "    finished",
    #     time=start,
    #     deep=f'added column: "{key_added}", sketch mask (bool) in `adata.obs` ',
    # )
    return adata if copy else None


def kernel_herding(X: np.ndarray, phi: np.ndarray, num_samples: int):
    """\
    Computes random fourier frequency features: https://arxiv.org/pdf/2207.00584.pdf

    Parameters
    X: np.ndarray
        array of input data (dimensions = cells x features)
    phi: np.ndarray
        random fourier features generated from the input data: `X`
    num_samples: int
        number of cells to subsample
    ----------
    Returns
    indices: np.ndarray
        indices of sampled cells (dimensions = (`num_samples`, ))
    subsample: np.ndarray
        subsampled `num_samples` cells (dimensions = `num_samples` x features)
    ----------

    """
    w_t = np.mean(phi, axis=0)
    w_0 = w_t
    indices, subsample = [], []
    ind_set = set()
    while len(indices) < num_samples:
        new_ind = np.argmax(np.dot(phi, w_t))
        x_t = X[new_ind]
        w_t = w_t + w_0 - phi[new_ind]
        if new_ind not in ind_set:
            indices.append(new_ind)
            subsample.append(x_t)
            ind_set.add(new_ind)

    indices = np.asarray(indices)
    subsample = np.stack(subsample, axis=0)

    return indices, subsample
