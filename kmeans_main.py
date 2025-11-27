import pandas as pd
import glob
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load all chunked results into a single DataFrame
all_files = sorted(glob.glob('data/kmeans_clustered/distributed_kmeans_chunk_*.csv'))
df_all = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)


# 2. Print cluster sizes
cluster_counts = Counter(df_all['kmeans_cluster'])
print("Distributed KMeans cluster sizes (top 10):", cluster_counts.most_common(10))
print("Total clusters:", len(cluster_counts))

# 3. Randomly sample for metric calculation (RAM-friendly)
sample_size = min(len(df_all), 50000)
sample_idx = np.random.choice(df_all.index, sample_size, replace=False)
features_for_clustering = ['LATITUDE', 'LONGITUDE', 'COUNCIL_DISTRICT', 'CENSUS_TRACT',
      'Bldg Type', 'Filing_Year', 'Filing_Month', 'Filing_Quarter', 'Permit_Age_Days', 'Permit Type_DM',
       'Permit Type_EQ', 'Permit Type_EW', 'Permit Type_FO', 'Permit Type_NB',
       'Permit Type_PL', 'Permit Type_SG', 'Job Type_A2', 'Job Type_A3',
       'Job Type_DM', 'Job Type_NB', 'Job Type_SG', 'Permit Status_ISSUED',
       'Permit Status_RE-ISSUED', 'Residential_YES']
X_sample = df_all.loc[sample_idx, features_for_clustering].astype(float).values
labels_sample = df_all.loc[sample_idx, 'kmeans_cluster']

if len(set(labels_sample)) > 1:
    sil_score = silhouette_score(X_sample, labels_sample)
    ch_score = calinski_harabasz_score(X_sample, labels_sample)
    db_score = davies_bouldin_score(X_sample, labels_sample)
    print("Distributed KMeans validation scores (sample):")
    print(f"  Silhouette Score: {sil_score:.3f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.1f}")
    print(f"  Davies-Bouldin Index: {db_score:.3f}")
else:
    print("Not enough clusters for quality scoring.")

# 4. Hotspot summary by cluster and month/year (optional, for reporting)
hotspot_summary = (
    df_all.groupby(['kmeans_cluster', 'Filing_Year', 'Filing_Month'])
    .agg(
        n_permits=('LATITUDE', 'size'),
        long_centroid=('LONGITUDE', 'median'),
        lat_centroid=('LATITUDE', 'median')
    )
    .reset_index()
)
print("\nHotspot summary table (top 10 rows):")
print(hotspot_summary.head(10))

# 5. (Optional) Save results for future analysis or mapping
df_all.to_csv('data/post_clustering/distributed_kmeans_full_results.csv', index=False)
hotspot_summary.to_csv('data/post_clustering/distributed_kmeans_hotspot_summary_full.csv', index=False)

# Assuming df_all is your concatenated results DataFrame with cluster assignments
# and original coordinates in 'LATITUDE_ORIG', 'LONGITUDE_ORIG'

import matplotlib.pyplot as plt
from collections import Counter

# Compute cluster sizes and sort by cluster size
cluster_counts = Counter(df_all['kmeans_cluster'])
clusters_by_size = sorted(cluster_counts.items(), key=lambda t: t[1], reverse=True)
cluster_order = [cl for cl, size in clusters_by_size]

plt.figure(figsize=(12, 12))
cmap = plt.get_cmap('tab10', len(cluster_order))

# Plot clusters in order: largest first, smallest (true hotspots) last!
for idx, cl in enumerate(cluster_order):
    pts = df_all[df_all['kmeans_cluster'] == cl]
    # Make smaller clusters more visible with larger points or darker color
    size_for_scatter = 8 if cluster_counts[cl] < 10000 else 2
    alpha = 1.0 if cluster_counts[cl] < 10000 else 0.05
    plt.scatter(
        pts['LONGITUDE_ORIG'],
        pts['LATITUDE_ORIG'],
        s=size_for_scatter,
        color=cmap(idx % cmap.N),
        alpha=alpha,
        label=f'Cluster {cl} (n={cluster_counts[cl]})' if cluster_counts[cl] < 10000 else None
    )

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('NYC Urban Development Hotspot Clusters (Hotspots Visible on Top)')
# Only show non-empty legend entries (i.e., just the hotspots)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize='small'
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('data/post_clustering/kmeans_hotspots_descending_full.png', dpi=150, bbox_inches='tight')
plt.close()
