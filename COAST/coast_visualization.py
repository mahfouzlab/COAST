# coast_visualization.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap


def run_umap_clustering(outdir, n_clusters, spot_size, background=True):
    """
    Compute UMAP embedding and KMeans clustering on concatenated _vit_mf CSVs
    and plot spatial clusters using _vit_coord CSVs.
    
    Args:
        outdir (str): Base output directory containing ViT_features/
        n_clusters (int): Number of clusters for KMeans
        spot_size (int): Dot size for scatterplots
        background (bool): If True, include all patches (intissue + grid).
                           If False, exclude patches ending in '_grid'.
    """

    vit_feature_dir = os.path.join(outdir, "ViT_features")
    vis_outdir = os.path.join(outdir, "visualization")
    os.makedirs(vis_outdir, exist_ok=True)

    # Get files
    vit_mf_files = sorted(glob.glob(os.path.join(vit_feature_dir, "*_vit_mf.csv")))
    vit_coord_files = sorted(glob.glob(os.path.join(vit_feature_dir, "*_vit_coord.csv")))

    if not vit_mf_files:
        raise FileNotFoundError(f"No _vit_mf.csv files found in {vit_feature_dir}")

    # Concatenate features
    dfs = []
    for f in vit_mf_files:
        df = pd.read_csv(f, index_col=0)
        if not background:  # remove grid rows
            df = df[~df.index.str.endswith("_grid")]
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    df_features = pd.concat(dfs, axis=0)
    print(f"[INFO] Concatenated {len(vit_mf_files)} ViT embedding files. Shape: {df_features.shape}")

    # Scale + UMAP
    feature_cols = [c for c in df_features.columns if c != "source_file"]
    features = StandardScaler().fit_transform(df_features[feature_cols])
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    df_features["UMAP1"], df_features["UMAP2"] = embedding[:, 0], embedding[:, 1]

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
    df_features["cluster"] = kmeans.labels_

    # Save combined dataset
    out_csv = os.path.join(vit_feature_dir, "umap_clusters_concatenated_sections.csv")
    df_features.to_csv(out_csv)
    print(f"[INFO] Dataset with UMAP + clusters saved in: {out_csv}")

    # --- UMAP plot by cluster ---
    plt.figure(figsize=(6, 5), dpi=200)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="cluster",
        palette="tab10", data=df_features,
        s=spot_size, linewidth=0
    )
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("UMAP1", fontsize=10)
    plt.ylabel("UMAP2", fontsize=10)
    plt.legend(markerscale=1.5, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_umap = os.path.join(vis_outdir, "umap_clusters.png")
    plt.savefig(out_umap, dpi=200)
    plt.show()
    print(f"[INFO] UMAP cluster plot saved in: {out_umap}")

    # --- UMAP plot colored by section ---
    section_colors = ["#F3CA40", "#5c8001"]  # yellow, green
    if "source_file" in df_features.columns:
        plt.figure(figsize=(6, 5), dpi=200)
        for i, src in enumerate(df_features["source_file"].unique()):
            df_sub = df_features[df_features["source_file"] == src]
            plt.scatter(
                df_sub["UMAP1"], df_sub["UMAP2"],
                c=section_colors[i % len(section_colors)],
                s=spot_size, alpha=0.7, label=os.path.splitext(src)[0],
                edgecolors="k", linewidth=0.01
            )
        plt.xlabel("UMAP1", fontsize=10)
        plt.ylabel("UMAP2", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(markerscale=1.5, fontsize=8, loc="upper right")
        plt.tight_layout()
        out_umap_section = os.path.join(vis_outdir, "umap_sections.png")
        plt.savefig(out_umap_section, dpi=200)
        plt.show()
        print(f"[INFO] UMAP plot colored by section saved in: {out_umap_section}")

    # --- Spatial plots ---
    if vit_coord_files:
        coord_dfs = []
        for f in vit_coord_files:
            df = pd.read_csv(f, index_col=0)
            if not background:
                df = df[~df.index.str.endswith("_grid")]
            df["source_file"] = os.path.basename(f)
            coord_dfs.append(df)
        df_coords = pd.concat(coord_dfs, axis=0)

        df_all = df_features.merge(df_coords, left_index=True, right_index=True, how="left")
        if "x" in df_all.columns and "y" in df_all.columns:
            unique_sources = df_all["source_file_y"].unique()
            n_plots = min(2, len(unique_sources))
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), dpi=300)
            if n_plots == 1:
                axes = [axes]
            for ax, src in zip(axes, unique_sources[:2]):
                df_sub = df_all[df_all["source_file_y"] == src]
                sns.scatterplot(
                    x="x", y=-df_sub["y"], hue="cluster",
                    palette="tab10", data=df_sub,
                    s=spot_size, linewidth=0, ax=ax
                )
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                ax.set_title(os.path.splitext(src)[0], fontsize=10)
                ax.legend(markerscale=1.5, fontsize=8, loc="upper right")
            plt.tight_layout()
            out_spatial = os.path.join(vis_outdir, "spatial_clusters.png")
            plt.savefig(out_spatial, dpi=300)
            plt.show()
            print(f"[INFO] Spatial cluster plots saved in: {out_spatial}")
    else:
        print("[WARNING] No coordinate files found -> skipping spatial plots")

    return df_features


def plot_final_aligned_scatter(outdir, spot_size, background=True):
    """
    Plot final aligned scatterplot of all *_aligned.csv files in CAST_output.

    Args:
        outdir (str): Base output directory (contains CAST_output/).
        spot_size (int): Size of scatterplot dots.
        background (bool): If True, include all patches (intissue + grid).
                           If False, exclude patches ending in '_grid'.
    """
    cast_outdir = os.path.join(outdir, "CAST_output")
    vis_outdir = os.path.join(outdir, "visualization")
    os.makedirs(vis_outdir, exist_ok=True)

    # Find all *_aligned.csv files
    aligned_files = sorted(glob.glob(os.path.join(cast_outdir, "*_aligned.csv")))
    if len(aligned_files) < 2:
        raise FileNotFoundError(f"Need at least 2 *_aligned.csv files in {cast_outdir}")

    plt.figure(figsize=(12, 12), dpi=200)

    # Pick a palette (yellow, green, blue, red, etc.)
    colors = ["#F3CA40", "#5c8001", "#1d3557", "#e63946", "#457b9d", "#ff7f0e"]

    for i, f in enumerate(aligned_files):
        df = pd.read_csv(f, index_col=0)

        if not background:
            df = df[~df.index.str.endswith("_grid")]

        if not {"x_aligned", "y_aligned"}.issubset(df.columns):
            print(f"[WARNING] Skipping {f} (missing x_aligned / y_aligned)")
            continue

        name = os.path.basename(f).replace("_aligned.csv", "")
        plt.scatter(
            df["x_aligned"], -df["y_aligned"],
            c=colors[i % len(colors)], s=spot_size, alpha=0.7,
            edgecolors="k", linewidth=0.01,
            label=name
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.legend(markerscale=1.5, fontsize=8, loc="upper right")
    plt.tight_layout()

    suffix = "" if background else "_nobg"
    out_final = os.path.join(vis_outdir, f"aligned_tissues{suffix}.png")
    plt.savefig(out_final, dpi=200)
    plt.show()
    print(f"[INFO] Final aligned scatterplot saved in: {out_final}")
