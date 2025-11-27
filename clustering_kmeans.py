from mpi4py import MPI
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load the preprocessed chunk (or use dfhotspot from previous step)
dfhotspot = pd.read_csv(f"data/processed/preprocessed_chunk_{rank}.csv")

# ---- 1. Define features for clustering ----
features_for_clustering = ['LATITUDE', 'LONGITUDE', 'COUNCIL_DISTRICT', 'CENSUS_TRACT',
      'Bldg Type', 'Filing_Year',
       'Filing_Month', 'Filing_Quarter', 'Permit_Age_Days', 'Permit Type_DM',
       'Permit Type_EQ', 'Permit Type_EW', 'Permit Type_FO', 'Permit Type_NB',
       'Permit Type_PL', 'Permit Type_SG', 'Job Type_A2', 'Job Type_A3',
       'Job Type_DM', 'Job Type_NB', 'Job Type_SG', 'Permit Status_ISSUED',
       'Permit Status_RE-ISSUED', 'Residential_YES']

# ---- 2. Impute missing temporal features ----
# (Only needed for features like 'PermitAgeDays')
# if 'PermitAgeDays' in dfhotspot.columns:
#     median_permit_age = dfhotspot['PermitAgeDays'].median()
#     dfhotspot['PermitAgeDays'] = dfhotspot['PermitAgeDays'].fillna(median_permit_age)

dfhotspot.dropna(inplace=True)

# ---- 3. Prepare feature matrix ----
X = dfhotspot[features_for_clustering].astype(float).values

X = X.astype(np.float32)

n_samples, n_features = X.shape

K = 10  # optimal cluster count from earlier analysis

if rank == 0:
    # Simple option: random K samples from master chunk
    initial_centroids = X[np.random.choice(X.shape[0], K, replace=False)]
else:
    initial_centroids = None

centroids = comm.bcast(initial_centroids, root=0)

MAX_ITERS = 150
tol = 1e-4  # convergence tolerance

for iteration in range(MAX_ITERS):
    # Assign each local point to closest centroid
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    # Compute local sums and counts
    local_sums = np.zeros((K, n_features))
    local_counts = np.zeros(K, dtype=int)
    for k in range(K):
        mask = (labels == k)
        if np.any(mask):
            local_sums[k] = X[mask].sum(axis=0)
            local_counts[k] = mask.sum()

    # Aggregate sums and counts at root
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)
    comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
    comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)

    # Update centroids at root, check for convergence
    if rank == 0:
        new_centroids = np.copy(centroids)
        for k in range(K):
            if global_counts[k] > 0:
                new_centroids[k] = global_sums[k] / global_counts[k]
        shift = np.linalg.norm(centroids - new_centroids)
        print(f"[ITER {iteration}] centroid shift: {shift:.6f}")
        converged = (shift < tol)
        centroids = new_centroids
    else:
        converged = None

    # Broadcast new centroids and convergence status
    centroids = comm.bcast(centroids, root=0)
    converged = comm.bcast(converged, root=0)

    if converged:
        break
    
final_dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
final_labels = np.argmin(final_dists, axis=1)
dfhotspot['kmeans_cluster'] = final_labels
dfhotspot.to_csv(f'data/kmeans_clustered/distributed_kmeans_chunk_{rank}.csv', index=False)



# # ---- 4. DBSCAN clustering ----
# n_clusters = 15
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# dfhotspot['kmeans_cluster'] = kmeans.fit_predict(X)

# # ---- 5. Mark hotspot points ----
# dfhotspot['ishotspot_k'] = dfhotspot['kmeans_cluster'] != -1

# # ---- 6. Save predictions for analysis/gathering ----
# dfhotspot.to_csv(f"data/kmeans_clustered/clustered_chunk_{rank}.csv", index=False)