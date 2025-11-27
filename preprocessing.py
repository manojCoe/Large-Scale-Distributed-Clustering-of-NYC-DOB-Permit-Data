from mpi4py import MPI
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # Should be 10

# ----- 1. Load your chunk -----
chunk_path = f"C:/Users/nandi/Downloads/DOB_Permit_data-20251116T004852Z-1-001/DOB_Permit_data/data_{rank}.csv"
df = pd.read_csv(chunk_path)

# ----- 2. Hotspot Columns -----
hotspot_cols = [
    'LATITUDE', 'LONGITUDE', 'COUNCIL_DISTRICT',
    'CENSUS_TRACT', 'Permit Type', 'Job Type',
    'Filing Date', 'Issuance Date', 'Bldg Type',
    'Residential', 'Permit Status'  # Adjust based on date fields in your data
]
dfhotspot = df[hotspot_cols].copy()

# ----- 3. Remove missing spatial values -----
dfhotspot = dfhotspot.dropna(subset=['LATITUDE', 'LONGITUDE'])

# ----- 4. Datetime Conversion -----
for date_col in ['Filing Date', 'Issuance Date']:
    if date_col in dfhotspot.columns:
        dfhotspot[date_col] = pd.to_datetime(dfhotspot[date_col], errors='coerce')

# ----- 5. Temporal Features -----
if 'Filing Date' in dfhotspot.columns:
    dfhotspot['Filing_Year'] = dfhotspot['Filing Date'].dt.year
    dfhotspot['Filing_Month'] = dfhotspot['Filing Date'].dt.month
    dfhotspot['Filing_Quarter'] = dfhotspot['Filing Date'].dt.quarter

if set(['Filing Date', 'Issuance Date']).issubset(dfhotspot.columns):
    dfhotspot['Permit_Age_Days'] = (dfhotspot['Issuance Date'] - dfhotspot['Filing Date']).dt.days

dfhotspot['Residential'].fillna("NO", inplace=True)

# ----- 6. Fill NAs, Reduce High Cardinality -----
# Categorical NA fill and cardinality reduction
for cat_col in ['Permit Type', 'Job Type', 'Residential', "Permit Status"]:
    if cat_col in dfhotspot.columns:
        dfhotspot[cat_col] = dfhotspot[cat_col].fillna('Unknown')
        if dfhotspot[cat_col].nunique() > 20:  # High cardinality: take top 19, rest as "Other"
            top_n = dfhotspot[cat_col].value_counts().nlargest(19).index
            dfhotspot[cat_col] = dfhotspot[cat_col].where(dfhotspot[cat_col].isin(top_n), 'Other')


# ----- 7. One-hot Encoding for Categoricals -----
dfhotspot = pd.get_dummies(dfhotspot, columns=['Permit Type', 'Job Type', 'Permit Status', 'Residential'], drop_first=True)

# ----- 8. Original Coordinates for Later -----
dfhotspot['LATITUDE_ORIG'] = dfhotspot['LATITUDE'].copy()
dfhotspot['LONGITUDE_ORIG'] = dfhotspot['LONGITUDE'].copy()

# ----- 9. Index Reset, Remove problem rows -----
dfhotspot.reset_index(drop=True, inplace=True)
dfhotspot.replace([np.inf, -np.inf], np.nan, inplace=True)
dfhotspot.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

# ----- 10. Numeric Scaling -----
numeric_cols = ['LATITUDE', 'LONGITUDE', 'COUNCIL_DISTRICT', 'CENSUS_TRACT', 'Bldg Type']
scaler = StandardScaler()
for col in numeric_cols:
    if col in dfhotspot.columns:
        values = dfhotspot[[col]].values
        dfhotspot[col] = scaler.fit_transform(values)

# Done preprocessing! Save your ready chunk if you like:
dfhotspot.to_csv(f"data/processed/preprocessed_chunk_{rank}.csv", index=False)
