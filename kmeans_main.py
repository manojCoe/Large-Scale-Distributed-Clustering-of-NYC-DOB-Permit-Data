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


# ============================================================
# CLUSTER INTERPRETATION
# ============================================================

print("\n" + "="*70)
print("CLUSTER INTERPRETATION")
print("="*70)

# Borough estimation from Council District
# Manhattan: 1-10, Bronx: 11-18, Brooklyn: 33-48, Queens: 19-32, Staten Island: 49-51
def estimate_borough(council_district):
    try:
        cd = int(council_district)
        if 1 <= cd <= 10:
            return 'Manhattan'
        elif 11 <= cd <= 18 or cd == 8:
            return 'Bronx'
        elif 19 <= cd <= 32:
            return 'Queens'
        elif 33 <= cd <= 48:
            return 'Brooklyn'
        elif 49 <= cd <= 51:
            return 'Staten Island'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

# One-hot column definitions
permit_type_cols = [c for c in df_all.columns if c.startswith('Permit Type_')]
job_type_cols = [c for c in df_all.columns if c.startswith('Job Type_')]
permit_status_cols = [c for c in df_all.columns if c.startswith('Permit Status_')]

# Permit Type full names
permit_type_names = {
    'DM': 'Demolition',
    'EQ': 'Equipment',
    'EW': 'Equipment Work',
    'FO': 'Foundation',
    'NB': 'New Building',
    'PL': 'Plumbing',
    'SG': 'Sign'
}

# Job Type full names
job_type_names = {
    'A1': 'Alteration Type 1',
    'A2': 'Alteration Type 2',
    'A3': 'Alteration Type 3',
    'DM': 'Demolition',
    'NB': 'New Building',
    'SG': 'Sign'
}

# Store results for table
interpretation_results = []

for cluster_id in sorted(df_all['kmeans_cluster'].unique()):
    cluster_data = df_all[df_all['kmeans_cluster'] == cluster_id]
    n_samples = len(cluster_data)
    pct = n_samples / len(df_all) * 100
    
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster_id}: {n_samples:,} samples ({pct:.1f}%)")
    print(f"{'='*50}")
    
    # ----- 1. Top Permit Type -----
    if permit_type_cols:
        permit_sums = cluster_data[permit_type_cols].sum()
        top_permit_col = permit_sums.idxmax()
        top_permit_code = top_permit_col.replace('Permit Type_', '')
        top_permit_name = permit_type_names.get(top_permit_code, top_permit_code)
        top_permit_pct = permit_sums[top_permit_col] / n_samples * 100
        print(f"  Top Permit Type: {top_permit_name} ({top_permit_code}) - {top_permit_pct:.1f}%")
    else:
        top_permit_name = 'N/A'
    
    # ----- 2. Top Job Type -----
    if job_type_cols:
        job_sums = cluster_data[job_type_cols].sum()
        top_job_col = job_sums.idxmax()
        top_job_code = top_job_col.replace('Job Type_', '')
        top_job_name = job_type_names.get(top_job_code, top_job_code)
        top_job_pct = job_sums[top_job_col] / n_samples * 100
        print(f"  Top Job Type: {top_job_name} ({top_job_code}) - {top_job_pct:.1f}%")
    else:
        top_job_name = 'N/A'
    
    # ----- 3. Residential Ratio -----
    if 'Residential_YES' in cluster_data.columns:
        residential_pct = cluster_data['Residential_YES'].mean() * 100
        residential_label = 'Residential' if residential_pct > 50 else 'Commercial/Mixed'
        print(f"  Residential: {residential_pct:.1f}% → {residential_label}")
    else:
        residential_pct = 0
        residential_label = 'Unknown'
    
    # ----- 4. Permit Status -----
    if permit_status_cols:
        status_sums = cluster_data[permit_status_cols].sum()
        top_status_col = status_sums.idxmax()
        top_status = top_status_col.replace('Permit Status_', '')
        top_status_pct = status_sums[top_status_col] / n_samples * 100
        print(f"  Top Permit Status: {top_status} - {top_status_pct:.1f}%")
    else:
        top_status = 'N/A'
    
    # ----- 5. Temporal Features -----
    if 'Filing_Year' in cluster_data.columns:
        avg_year = cluster_data['Filing_Year'].mean()
        print(f"  Avg Filing Year: {avg_year:.1f}")
    else:
        avg_year = 0
    
    if 'Permit_Age_Days' in cluster_data.columns:
        avg_age = cluster_data['Permit_Age_Days'].mean()
        print(f"  Avg Permit Processing Time: {avg_age:.1f} days")
    else:
        avg_age = 0
    
    # ----- 6. Geographic Center -----
    if 'LATITUDE_ORIG' in cluster_data.columns and 'LONGITUDE_ORIG' in cluster_data.columns:
        lat_center = cluster_data['LATITUDE_ORIG'].mean()
        long_center = cluster_data['LONGITUDE_ORIG'].mean()
        print(f"  Geographic Center: ({lat_center:.4f}, {long_center:.4f})")
    else:
        lat_center, long_center = 0, 0
    
    # ----- 7. Estimate Borough from Council District -----
    if 'COUNCIL_DISTRICT' in cluster_data.columns:
        # Unscale if needed (assuming it was standardized)
        # For now, use mode of rounded values
        try:
            borough_counts = cluster_data['COUNCIL_DISTRICT'].apply(
                lambda x: estimate_borough(round(x * 10 + 25))  # rough unscaling
            ).value_counts()
            top_borough = borough_counts.index[0] if len(borough_counts) > 0 else 'Unknown'
        except:
            top_borough = 'Unknown'
        print(f"  Estimated Borough: {top_borough}")
    else:
        top_borough = 'Unknown'
    
    # ----- 8. Generate Cluster Label -----
    # Create a human-readable cluster description
    if n_samples < 1000:
        size_label = "Specialized/Rare"
    elif pct < 5:
        size_label = "Small"
    elif pct < 20:
        size_label = "Medium"
    else:
        size_label = "Large"
    
    cluster_label = f"{size_label} {residential_label} - {top_permit_name}"
    print(f"\n  → CLUSTER LABEL: {cluster_label}")
    
    # Store for summary table
    interpretation_results.append({
        'Cluster': cluster_id,
        'Size': n_samples,
        'Pct': f"{pct:.1f}%",
        'Top_Permit': top_permit_name,
        'Top_Job': top_job_name,
        'Residential%': f"{residential_pct:.1f}%",
        'Avg_Year': f"{avg_year:.0f}" if avg_year > 0 else 'N/A',
        'Processing_Days': f"{avg_age:.0f}" if avg_age > 0 else 'N/A',
        'Label': cluster_label
    })

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("CLUSTER INTERPRETATION SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame(interpretation_results)
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('data/cluster_interpretation.csv', index=False)
print("\n✓ Saved: data/cluster_interpretation.csv")

# ============================================================
# MARKDOWN TABLE FOR PRESENTATION
# ============================================================
print("\n" + "="*70)
print("MARKDOWN TABLE (Copy for Presentation)")
print("="*70)

print("\n| Cluster | Size | % | Top Permit | Top Job | Residential | Label |")
print("|---------|------|---|------------|---------|-------------|-------|")
for r in interpretation_results:
    print(f"| {r['Cluster']} | {r['Size']:,} | {r['Pct']} | {r['Top_Permit']} | {r['Top_Job']} | {r['Residential%']} | {r['Label']} |")

print("\n" + "="*70)
print("CLUSTER INTERPRETATION COMPLETE")
print("="*70)