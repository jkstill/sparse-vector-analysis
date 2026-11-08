#!/usr/bin/env python3

"""
Group similar SQL statements from a file to characterize a workload.

Input: one SQL statement per line (trailing whitespace is ignored), optionally
prefixed with an execution count: <execution_count>:<sql statement>
(a tab may be used instead of the colon, as produced by mrskew).
Lines without a prefix count as 1 execution.

Each statement is normalized (string/numeric literals and bind variables removed,
whitespace collapsed), converted to a TF-IDF sparse vector using the same
SQL-aware tokenizer as spv-sql.py (minus 1-char aliases), and clustered with agglomerative
clustering on cosine distance.
"""

import argparse
import csv
import re
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('sqlfile', help='file with one SQL statement per line, optionally <count>:<sql> or <count><TAB><sql>')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='max cosine distance for merging groups (0.3 = similarity >= 0.7). default: %(default)s')
parser.add_argument('--top-features', type=int, default=8, help='features shown per group. default: %(default)s')
parser.add_argument('--no-keywords', action='store_true',
                    help='ignore SQL keywords so groups reflect tables/columns only')
parser.add_argument('--csv', help='optional output file: one row per statement with its group id')
args = parser.parse_args()


def normalize(sql):
    sql = re.sub(r"'(?:[^']|'')*'", ' ', sql)   # string literals
    sql = re.sub(r':\w+', ' ', sql)              # bind variables
    sql = re.sub(r'\b\d+(?:\.\d+)?\b', '0', sql)  # numeric literals (identifiers like F1 are untouched)
    return ' '.join(sql.split())


# Used with --no-keywords. Double-quoted identifiers are never removed, so columns
# such as "Date" or "Type" survive.
SQL_KEYWORDS = set('''
select from where and or not in is null like between exists any all some as on using
join inner outer left right full cross natural group by order having distinct unique
asc desc nulls first last union intersect minus except with case when then else end
insert into values update set delete merge matched begin declare commit rollback
alter session table index view create drop truncate for of nowait wait skip locked
rownum rowid level connect start prior fetch next rows only offset limit escape
count sum avg min max nvl nvl2 coalesce decode upper lower trim rtrim ltrim substr
instr length to_char to_date to_number trunc round sysdate systimestamp cast
'''.split())


def drop_keywords(sql):
    return re.sub(r'"[^"]*"|\b[A-Za-z_]\w*\b',
                  lambda m: ' ' if m.group(0).lower() in SQL_KEYWORDS else m.group(0), sql)


# 1. Read statements; identical normalized statements are collapsed and counted
raw, execs = [], []
with open(args.sqlfile, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(\d+)[:\t](.*)', line)  # SQL never starts with a digit, so this is unambiguous
        execs.append(int(m.group(1)) if m else 1)
        raw.append(m.group(2).strip() if m else line)

norm = [normalize(s) for s in raw]
counts = Counter()
for n, e in zip(norm, execs):
    counts[n] += e
unique = list(counts)
example = {}
for r, n in zip(raw, norm):
    example.setdefault(n, r)

print(f"Statements read: {len(raw)}  Executions: {sum(execs)}  Unique after normalization: {len(unique)}")

# 2. Sparse TF-IDF vectors. Same tokenizer as spv-sql.py, except single-character
#    tokens (typically table aliases such as A, P, L) are dropped as noise.
vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_\.]+\b')
X = vectorizer.fit_transform([drop_keywords(u) for u in unique] if args.no_keywords else unique)
feature_names = vectorizer.get_feature_names_out()
print(f"Dictionary size: {len(feature_names)}  Matrix shape: {X.shape}")

# 3. Cluster on cosine distance; the threshold decides how many groups emerge
labels = AgglomerativeClustering(
    n_clusters=None, metric='precomputed', linkage='average', distance_threshold=args.threshold
).fit_predict(cosine_distances(X))

# 4. Summarize groups, ordered by share of total executions
weights = np.array([counts[u] for u in unique])
total = weights.sum()
groups = sorted(set(labels), key=lambda g: -weights[labels == g].sum())
print(f"Groups found: {len(groups)} (threshold {args.threshold})\n")

for rank, g in enumerate(groups, 1):
    idx = np.where(labels == g)[0]
    centroid = np.asarray(X[idx].mean(axis=0)).ravel()
    rep = idx[np.argmax(X[idx] @ centroid)]  # statement closest to the group centroid
    top = [feature_names[i] for i in np.argsort(centroid)[::-1][:args.top_features] if centroid[i] > 0]
    n_exec = weights[idx].sum()
    print(f"=== Group {rank}: {n_exec} executions ({100 * n_exec / total:.1f}%), {len(idx)} distinct")
    print(f"    Features : {', '.join(top)}")
    print(f"    Example  : {example[unique[rep]][:300]}\n")

# 5. Optional per-statement assignment
if args.csv:
    rank_of = {g: r for r, g in enumerate(groups, 1)}
    label_of = dict(zip(unique, labels))
    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'executions', 'sql'])
        for r, n, e in zip(raw, norm, execs):
            w.writerow([rank_of[label_of[n]], e, r])
    print(f"Assignments written to {args.csv}")
