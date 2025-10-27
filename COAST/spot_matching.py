import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import scanpy as sc
from scipy.sparse import issparse
from tqdm import tqdm


def load_data(data_path, coord_dir=None, mode=None):
    """
    Load molecular or ViT features and corresponding coordinates.
    Returns a pandas DataFrame with features (rows = spot IDs) and optional x,y columns.
    """
    if os.path.isdir(data_path):
        raise ValueError(
            f"data_path is a directory ({data_path}). "
            "If you want to pair directories of molecular + coord files, use pair_and_load()."
        )

    # Load features
    if data_path.endswith(".csv"):
        df_features = pd.read_csv(data_path, index_col=0)
    elif data_path.endswith(".h5ad"):
        adata = sc.read_h5ad(data_path)
        X = adata.X.todense() if issparse(adata.X) else adata.X
        df_features = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    elif data_path.endswith(".h5"):
        adata = sc.read_10x_h5(data_path)
        X = adata.X.todense() if issparse(adata.X) else adata.X
        df_features = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    # Load coordinates (optional)
    if coord_dir is not None:
        if os.path.isfile(coord_dir):
            coord_file = coord_dir
        else:
            basename = os.path.splitext(os.path.basename(data_path))[0]
            allowed_ext = (".csv", ".h5", ".h5ad")
            candidates = [f"{basename}_vit_coord.csv", f"{basename}.csv"]
            coord_file = None

            for name in candidates:
                p = os.path.join(coord_dir, name)
                if os.path.exists(p):
                    coord_file = p
                    break

            if coord_file is None:
                csvs = sorted([
                    os.path.join(coord_dir, f)
                    for f in os.listdir(coord_dir)
                    if not f.startswith(".") and (os.path.isdir(os.path.join(coord_dir, f)) or f.endswith(allowed_ext))
                ])
                for cf in csvs:
                    if basename in os.path.splitext(os.path.basename(cf))[0]:
                        coord_file = cf
                        break
                if coord_file is None:
                    if len(csvs) == 1:
                        coord_file = csvs[0]
                    elif len(csvs) > 1:
                        print(f"Multiple files in coord dir '{coord_dir}', none matched '{basename}'. Using first: {os.path.basename(csvs[0])}")
                        coord_file = csvs[0]
                    else:
                        raise FileNotFoundError(f"No CSV/H5/H5AD coordinate files found in directory: {coord_dir}")

        df_coords = pd.read_csv(coord_file, index_col=0)
        if not {"x", "y"}.issubset(df_coords.columns):
            raise ValueError(f"Coordinate file {coord_file} missing 'x' and 'y' columns")
        df = df_features.merge(df_coords[['x', 'y']], left_index=True, right_index=True, how="left")
    else:
        df = df_features

    # Filter by mode (moldata = indices ending with _intissue; nomoldata = _grid)
    if mode in ['moldata', 'nomoldata']:
        suffix = '_intissue' if mode == 'moldata' else '_grid'
        filtered_idx = [idx for idx in df.index if str(idx).endswith(suffix)]
        if filtered_idx:
            df = df.loc[filtered_idx]

    # Cast coordinates safely
    if {'x', 'y'}.issubset(df.columns):
        if df[['x', 'y']].isna().any().any():
            df[['x', 'y']] = df[['x', 'y']].astype(float)
        else:
            df[['x', 'y']] = df[['x', 'y']].astype(int)

    return df


def pair_and_load(mol_dir, coord_dir, mode=None):
    """
    Pair molecular and coordinate files by sorted order and load them using load_data.
    Returns list of DataFrames (one per pair).
    """
    allowed_ext = (".csv", ".h5ad", ".h5")
    mol_files = sorted([os.path.join(mol_dir, f) for f in os.listdir(mol_dir)
                        if not f.startswith(".") and f.endswith(allowed_ext)])
    coord_paths = sorted([os.path.join(coord_dir, it)
                          for it in os.listdir(coord_dir)
                          if not it.startswith(".") and (os.path.isdir(os.path.join(coord_dir, it)) or it.endswith(allowed_ext))])

    n_pairs = min(len(mol_files), len(coord_paths))
    if len(mol_files) != len(coord_paths):
        print(f"Molecular data files: {len(mol_files)}, coord items: {len(coord_paths)} — pairing first {n_pairs} after sorting.")

    dfs = []
    for i in range(n_pairs):
        mol_path = mol_files[i]
        coord_path = coord_paths[i]
        print(f"{os.path.basename(mol_path)} (molecular data) is paired with {os.path.basename(coord_path)} (coord)")
        df = load_data(mol_path, coord_dir=coord_path, mode=mode)
        dfs.append(df)

    return dfs


def match_spots(outdir, max_distance=90):
    """
    Match spots between two CAST-aligned tissues using KD-tree.
    Returns ccf_ref with columns: query_idx, ref_idx, x_aligned, y_aligned (from tissue1).
    """
    CAST_DIR = os.path.join(outdir, "CAST_output")
    cast_files = sorted([f for f in glob.glob(os.path.join(CAST_DIR, "*.csv")) if not os.path.basename(f).startswith(".")])
    if len(cast_files) < 2:
        raise FileNotFoundError(f"Expected at least 2 CAST-aligned CSVs in {CAST_DIR}, found {len(cast_files)}")

    tissue1_ccf = pd.read_csv(cast_files[0], index_col=0)
    tissue2_ccf = pd.read_csv(cast_files[1], index_col=0)

    # Build KD-tree from tissue2
    query_coords = tissue2_ccf[['x_aligned', 'y_aligned']].to_numpy()
    tree = cKDTree(query_coords)
    ref_coords = tissue1_ccf[['x_aligned', 'y_aligned']].to_numpy()
    distances, indices = tree.query(ref_coords, distance_upper_bound=max_distance)

    ccf_query_index = tissue2_ccf.index.to_numpy()
    matched_indices = [ccf_query_index[i] if dist <= max_distance else pd.NA
                       for i, dist in zip(indices, distances)]

    ccf_ref = tissue1_ccf.copy()
    # Append the matched query index (may contain pd.NA for no-match)
    ccf_ref['query_idx'] = matched_indices
    # Also keep the reference index as a column (original index)
    ccf_ref['ref_idx'] = ccf_ref.index

    # Return only matched rows (same as your original)
    return ccf_ref.dropna(subset=['query_idx']).copy()


def build_multimodal_anndata(outdir, mol_dir, coord_dir, metadata=None, max_distance=90):
    """
    Build a multimodal AnnData from two tissues using molecular/ViT data and CAST-aligned coordinates.
    Produces identical output to your original pipeline but optimized for memory.

    Returns:
        AnnData object (also saved to outdir/multimodal_anndata/multimodal_dataset.h5ad)
    """
    CAST_DIR = os.path.join(outdir, "CAST_output")
    MULTIMODAL_OUTDIR = os.path.join(outdir, "multimodal_anndata")
    os.makedirs(MULTIMODAL_OUTDIR, exist_ok=True)

    # --- Load CAST-aligned CSVs (these will be used for coordinate queries) ---
    cast_files = sorted([f for f in glob.glob(os.path.join(CAST_DIR, "*.csv"))
                         if not os.path.basename(f).startswith(".")])
    if len(cast_files) < 2:
        raise FileNotFoundError(f"Expected at least 2 CAST-aligned CSVs in {CAST_DIR}, found {len(cast_files)}")

    tissue1_ccf = pd.read_csv(cast_files[0], index_col=0)
    tissue2_ccf = pd.read_csv(cast_files[1], index_col=0)

    # Remove grid spots from CAST ccf indexes (as in original)
    tissue1_ccf = tissue1_ccf[~tissue1_ccf.index.str.endswith('_grid')]
    tissue2_ccf = tissue2_ccf[~tissue2_ccf.index.str.endswith('_grid')]

    # Keep consistent naming (strip trailing _intissue so lookups match)
    tissue1_ccf.index = tissue1_ccf.index.str.replace('_intissue$', '', regex=True)
    tissue2_ccf.index = tissue2_ccf.index.str.replace('_intissue$', '', regex=True)

    # --- Load molecular/ViT data (pair and load using your function) ---
    dfs = pair_and_load(mol_dir, coord_dir, mode='moldata')
    if len(dfs) < 2:
        raise RuntimeError("Expected at least 2 paired dataframes from pair_and_load()")
    tissue1_df, tissue2_df = dfs[0], dfs[1]

    # Ensure molecular data indices are cleaned to match CAST-cleaned indices:
    tissue1_df.index = tissue1_df.index.astype(str).str.replace('_intissue$', '', regex=True)
    tissue2_df.index = tissue2_df.index.astype(str).str.replace('_intissue$', '', regex=True)

    # --- Match spots using KD-tree (get mapping between tissue1 (ref) and tissue2 (query)) ---
    ccf_ref = match_spots(outdir, max_distance=max_distance)

    # Remove rows that correspond to grid indices (same as your original code)
    if 'ref_idx' in ccf_ref.columns:
        ccf_ref = ccf_ref[~ccf_ref['ref_idx'].astype(str).str.endswith('_grid')]
    if 'query_idx' in ccf_ref.columns:
        ccf_ref = ccf_ref[~ccf_ref['query_idx'].astype(str).str.endswith('_grid')]

    # Create cleaned index strings without the _intissue suffix for lookups
    ccf_ref['ref_idx_clean'] = ccf_ref['ref_idx'].astype(str).str.replace('_intissue$', '', regex=True)
    ccf_ref['query_idx_clean'] = ccf_ref['query_idx'].astype(str).str.replace('_intissue$', '', regex=True)

    # --- Build multimodal feature matrix in a memory-efficient, vectorized way ---
    ref_list = ccf_ref['ref_idx_clean'].tolist()
    query_list = ccf_ref['query_idx_clean'].tolist()
    n_spots = len(ref_list)
    if n_spots == 0:
        raise RuntimeError("No matched spots found after filtering; nothing to build.")

    # Vectorized selection: returns DataFrames in the same order as ref_list/query_list
    mat1_df = tissue1_df.loc[ref_list]
    mat2_df = tissue2_df.loc[query_list]

    # Convert to numpy arrays (no copies if possible)
    mat1 = mat1_df.to_numpy(copy=False)
    mat2 = mat2_df.to_numpy(copy=False)

    # Concatenate horizontally to create feature matrix X with same column order you had
    X = np.hstack([mat1, mat2])

    # Build var names in the same order used before
    var_names = [f"{c}_tissue1" for c in tissue1_df.columns] + [f"{c}_tissue2" for c in tissue2_df.columns]

    # Build obs index identical to your prior code
    obs_index = [f"Spot_{i+1}" for i in range(n_spots)]

    # --- Construct AnnData directly from X to avoid intermediate huge DataFrames ---
    adata = sc.AnnData(X=X)
    adata.obs = pd.DataFrame(index=obs_index)
    adata.var = pd.DataFrame(index=var_names)

    # Add tissue information to var (same logic as before)
    adata.var['tissue'] = [
        'tissue1' if name.endswith('_tissue1') else
        'tissue2' if name.endswith('_tissue2') else 'unknown'
        for name in adata.var.index
    ]

    # --- Metadata mapping (vectorized) ---
    if metadata is not None:
        # Load metadata frames
        meta1_df = None
        meta2_df = None
        if isinstance(metadata, str):
            if os.path.isdir(metadata):
                meta_files = sorted([os.path.join(metadata, f) for f in os.listdir(metadata)
                                     if f.lower().endswith('.csv')])
                if len(meta_files) != 2:
                    raise ValueError("Metadata directory must contain exactly 2 CSV files for tissues.")
                meta1_df = pd.read_csv(meta_files[0], index_col=0)
                meta2_df = pd.read_csv(meta_files[1], index_col=0)
            else:
                meta1_df = pd.read_csv(metadata, index_col=0)
        elif isinstance(metadata, sc.AnnData):
            meta1_df = metadata.obs.copy()
        else:
            raise ValueError("metadata must be a CSV path, directory, or AnnData object")

        meta_features = {}
        # Tissue1 metadata: reindex by ref_list to preserve ccf_ref order (inserts NaN for missing)
        if meta1_df is not None:
            meta1_reindexed = meta1_df.reindex(ref_list)
            for col in meta1_reindexed.columns:
                meta_features[f"{col}_tissue1"] = meta1_reindexed[col].values

        # Tissue2 metadata: reindex by query_list
        if meta2_df is not None:
            meta2_reindexed = meta2_df.reindex(query_list)
            for col in meta2_reindexed.columns:
                meta_features[f"{col}_tissue2"] = meta2_reindexed[col].values

        if meta_features:
            meta_df = pd.DataFrame(meta_features, index=obs_index)
            adata.obs = pd.concat([adata.obs, meta_df], axis=1)
            
    adata = adata[:, ~adata.var_names.str.contains(r'^(x|y)_tissue\d+$', case=False)].copy()


    # --- Save AnnData (same path as your original) ---
    out_h5ad = os.path.join(MULTIMODAL_OUTDIR, "multimodal_dataset.h5ad")
    adata.write(out_h5ad)
    print(f"Multimodal AnnData saved to {out_h5ad}, shape={adata.shape}")

    return adata


# # If you want a small test harness (example):
# if __name__ == "__main__":
#     # Example usage (adjust paths to your environment)
#     # outdir = "/path/to/outdir"
#     # mol_dir = "/path/to/molecular_dir"
#     # coord_dir = "/path/to/coord_dir"
#     # ad = build_multimodal_anndata(outdir, mol_dir, coord_dir, metadata=None, max_distance=90)
#     pass
