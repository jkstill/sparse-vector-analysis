#!/usr/bin/env python3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Sample workload: A mix of Reporting, Point Lookups, and a potentially anomalous query
sql_queries = [
    # Query 0: Standard daily sales aggregation report
    "SELECT COUNT(o.order_id), SUM(o.total_amount) FROM orders o JOIN customers c ON o.cust_id = c.id WHERE o.status = 'COMPLETED' GROUP BY c.region",
    
    # Query 1: Similar report, slightly different columns but identical structural pattern
    "SELECT COUNT(o.order_id), AVG(o.total_amount) FROM orders o JOIN customers c ON o.cust_id = c.id WHERE o.status = 'PENDING' GROUP BY c.country",
    
    # Query 2: Simple transactional point lookup by primary key
    "SELECT name, email FROM customers WHERE id = 4552",
    
    # Query 3: Potential anomalous extraction or a rare security audit query
    "SELECT * FROM system_logs WHERE event_type = 'LOGIN_FAIL' AND severity = 'CRITICAL' ORDER BY timestamp DESC"
]

# 2. Configure a TfidfVectorizer tailored for SQL syntax.
# The custom token_pattern ensures table/column names with dots or underscores (e.g., 'o.order_id') 
# are preserved as distinct, atomic features rather than being split on punctuation.
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_\.]*\b'
)

# 3. Fit the dictionary and transform the SQL queries into sparse vectors
# This calculates Term Frequency - Inverse Workload Frequency (TF-IWF)
sparse_matrix = vectorizer.fit_transform(sql_queries)

# 4. Inspect the resulting high-dimensional sparse feature set
feature_names = vectorizer.get_feature_names_out()
print(f"Global Workload Dictionary Size: {len(feature_names)} unique dimensions.")
print(f"Sparse Matrix Internal Shape: {sparse_matrix.shape} (Queries x Features)\n")

# 5. Evaluate workload similarity using Cosine Similarity
similarity_matrix = cosine_similarity(sparse_matrix)

print("--- Workload Similarity Analysis ---")
print(f"Similarity between Reporting Query 0 and Reporting Query 1: {similarity_matrix[0][1]:.4f} (High Similarity)")
print(f"Similarity between Reporting Query 0 and Point Lookup Query 2: {similarity_matrix[0][2]:.4f} (Low Similarity)")
print(f"Similarity between Reporting Query 0 and Audit/Anomaly Query 3 : {similarity_matrix[0][3]:.4f} (Near Zero Similarity)")

# 6. Interpretability Check: Show which dimensions heavily influenced the similarity between Query 0 and 1
q0_vector = sparse_matrix.toarray()[0]
q1_vector = sparse_matrix.toarray()[1]
shared_features = np.minimum(q0_vector, q1_vector)

print("\n--- Shared High-Weight Features (Query 0 & Query 1) ---")
for idx in np.argsort(shared_features)[::-1]:
    if shared_features[idx] > 0:
        print(f" Dimension Match: '{feature_names[idx]}' (Weight contribution: {shared_features[idx]:.4f})")

