# Multi-Modal Consecutive Tissue Alignment
##### Author: Benedetta Manzato

This repository contains a Python script that processes high-resolution tissue images and performs image patch extraction, feature extraction using ViT (Vision Transformer), CAST-based alignment, and K-Means clustering of the extracted features. The workflow integrates spatial transcriptomics data and image data for tissue analysis.

## Features

- **Image Patch Extraction**: Extracts patches from input images based on centroid coordinates and uses a pre-trained Vision Transformer (ViT) model for feature extraction.
- **CAST Alignment**: Uses the CAST algorithm to align the extracted features spatially and apply data embedding techniques.
- **K-Means Clustering**: Performs K-Means clustering on the extracted features and visualizes the clustered data.
- **Visualization**: Generates high-quality plots to visualize the alignment and clustering results.

## Requirements

- Python 3.8+
- PyTorch
- Scanpy
- CAST
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm
- Pillow

### Install Dependencies

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Usage
```bash
python script.py \
  --img_dir /path/to/image1.tiff /path/to/image2.tiff \
  --centroid_dir /path/to/centroid1.csv /path/to/centroid2.csv \
  --output_dir /path/to/output_directory \
  --step_size 32
```

#### Command-line Arguments
--img_dir: Paths to two TIFF image files. These should be high-resolution images for which you want to extract patches.
--centroid_dir: Paths to two centroid files (CSV or AnnData). These files should contain the coordinates (x, y) of the centroids for patch extraction.
--output_dir: Directory where the output files (such as extracted features, alignment data, and plots) will be saved.
--step_size: The step size for sampling outside points when extracting image patches (default is 32).

#### Example command
```bash
python script.py \
  --img_dir /path/to/image1.tiff /path/to/image2.tiff \
  --centroid_dir /path/to/centroid1.csv /path/to/centroid2.csv \
  --output_dir /path/to/output_directory \
  --step_size 32
```

### Workflow Overview
Step 1: Image Patch Extraction
What it does:

The script extracts image patches from the provided TIFF image files. It uses the centroid coordinates from the provided centroid_dir (CSV or AnnData files) to define the areas from which to extract patches. The script extracts patches both around the centroids and from random positions outside the centroids using the step_size parameter to define the spacing of the outside patches.
How: It uses the pre-trained Vision Transformer (ViT) model to extract features from these patches.
Input:

Two image files and their corresponding centroid files.
Output:

ViT-based features are saved as CSV files under the ViT_MF_{img_name}.csv.
Coordinates of the extracted patches are saved in ViT_coord_{img_name}.csv.
Step 2: CAST Alignment
What it does:
The extracted features are then aligned using the CAST algorithm, which processes the data through various embedding and alignment techniques.
It first reads the extracted features and coordinates from the previous step, then applies spatial alignment and embedding methods to match the features from the two images based on the centroids.
Input:
Extracted features and coordinates from the image patches.
Output:
Final aligned coordinates for each sample saved as CSV files, named coord_final_{sample}.csv.
Step 3: K-Means Clustering
What it does:

After alignment, the script applies K-Means clustering on the features to categorize them into different groups. The K-Means algorithm groups the features into clusters and assigns a label to each data point.
The script visualizes the clustering results using scatter plots, with each cluster having a distinct color.
Input:

Extracted features for the patches.
Output:

Cluster labels saved as kmeans_ViT.csv.
Step 4: Visualization
What it does:
The script generates two primary plots:
K-Means Clustering Plot: Displays the K-Means clustering results, showing how the patches are distributed across the clusters.
Final Alignment Plot: Displays the final alignment of the two image samples after CAST processing.
Input:
Aligned coordinates and K-Means cluster data.
Output:
High-quality PNG plots, saved under the figures/ directory:
ViT_embedding_kmeans_plot.png for K-Means clustering.
final_alignment.png for final alignment.


### Output
The script will generate the following output files in the specified --output_dir:

ViT Features: CSV files containing the extracted features for each image (ViT_MF_{img_name}.csv).
Final Alignment Coordinates: CSV files with the aligned coordinates of the image samples (coord_final_{sample}.csv).
K-Means Clustering: CSV files containing the K-Means cluster assignments (kmeans_ViT.csv).
Figures: High-resolution PNG plots visualizing the clustering and alignment results, saved in the figures/ subdirectory.

#### Plotting Results
K-Means Clustering Plot: Displays the spatial distribution of K-Means clusters from the extracted features. The clusters are visualized using different colors.
Final Alignment Plot: Displays the final spatial alignment of the two image samples after CAST processing.
The plots are saved as PNG images in the figures/ directory, and the alignment results are saved as CSV files.






### Citation 
- ViT
- CAST

#### Contact
b.manzato@lumc.nl
