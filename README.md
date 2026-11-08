# Sparse Vector Analysis of SQL Workloads

Group similar SQL statements together to characterize a database workload.
Each statement is turned into a sparse TF-IDF vector of its words (tables,
columns, keywords), and statements whose vectors point in a similar direction
(cosine similarity) are placed in the same group.

## Files

- `spv-sql.py` - small demo of the technique on four hard-coded statements.
- `spv-cluster.py` - reads SQL statements from a file, groups them and reports
  the groups ordered by share of executions.

## Requirements

Python 3 with `numpy` and `scikit-learn`. Developed and tested with
scikit-learn 1.9.1 and numpy 2.5.1.

The scripts use `#!/usr/bin/env python3`, so the first `python3` on the PATH
must have these modules. If it does not, activate an environment that does,
or run the script with that interpreter directly:

```bash
~/miniconda3/envs/py312-base/bin/python spv-cluster.py sql_text.txt
```

## SQL Text Acquisition

Ideally a text file containing the number of executions and the SQL text is
used to drive the analysis (see Input File Format below).

I am using Method R tools to extract this information from Oracle SQL (10046)
trace files. Only top level SQL (`dep==0`) is included.

```bash
mrskew --nohead --nofoot --nodashes --nocommas \
    --sep=$'\t' --top=0 --name='^EXEC$' \
    --group='$sql' --gl='SQL_TEXT' \
    --select='1' --sl='EXECS' --sort=2nd \
    --where='$dep==0' *.trc \
    | perl -F'\t' -lane '$F[0]=~s/\s+$//; printf "%d\t%s\n", $F[1], $F[0]' \
    > sql_text.txt
```

## Input File Format

One SQL statement per line. Leading and trailing whitespace is ignored, as
are blank lines. Each line may start with an execution count:

```
<execution_count><TAB><sql statement>
<execution_count>:<sql statement>
<sql statement>
```

- The separator may be a tab (as produced by the `mrskew` command above) or a
  colon.
- A line without a count is counted as 1 execution. A file of verbatim SQL
  taken from trace files, one line per execution, works as is.
- Formats may be mixed in one file. Since SQL never starts with a digit, a
  leading number is always read as the count.
- Multi-line statements are not supported; each statement must be on a
  single line. Statements separated by `;` or `/` must be converted first.

Examples:

```
303	SELECT L."Description" FROM LANGCD L, PATIENT P WHERE P."Patient" = 109341 AND P."PrefLanguage" = L."LangCd"
127:SELECT "WaitList", "Appointment" FROM WAITLIST WHERE "Appointment" = 'NRML'
SELECT * FROM FITEM WHERE "FItem" = 316839
```

## Usage

```
spv-cluster.py [--threshold 0.3] [--top-features 8] [--no-keywords] [--min-execs 1] [--csv FILE] sqlfile
```

| Option | Meaning |
|---|---|
| `--threshold` | Maximum cosine distance for merging groups. Default 0.3 (similarity of at least 0.7). See below. |
| `--top-features` | Number of highest weighted words shown per group. Default 8. |
| `--no-keywords` | Ignore SQL keywords and common functions, so groups reflect tables and columns only. |
| `--min-execs N` | Only report groups with at least N total executions. Default 1 (all groups). Grouping and the CSV are not affected, and group numbers stay the same. |
| `--csv FILE` | Also write one row per input line: `group,executions,sql`. |

Example output:

```
Statements read: 5691  Executions: 5691  Unique after normalization: 962
Dictionary size: 2261  Matrix shape: (962, 2261)
Groups found: 611 (threshold 0.3)
Minimum executions shown: 1 (0 groups with 0 executions (0.0%) not shown)

=== Group 1: 303 executions (5.3%), 1 distinct
    Features : langcd, preflanguage, patient, description, and, select, where, from
    Example  : SELECT L."Description" FROM LANGCD L, PATIENT P WHERE P."Patient" = 109341 AND P."PrefLanguage" = L."LangCd"
```

- **Minimum executions shown**: the `--min-execs` setting, and how many groups
  and executions fell below it. With `--min-execs 50` on the sample file, 21
  groups are shown and 590 groups holding 63.9% of executions are not.
- **executions**: total executions of all statements in the group, and the
  percentage of all executions in the file.
- **distinct**: number of distinct statement patterns (after normalization)
  in the group.
- **Features**: the words with the highest average TF-IDF weight in the group.
- **Example**: the statement closest to the center of the group, as it
  appeared in the input.

## How It Works

1. **Normalize.** String literals, numeric literals and bind variables are
   removed and whitespace is collapsed. Statements that differ only in
   literal values become identical, and their executions are added together.
   In the sample file, 5691 statements reduce to 962 distinct patterns.
   Identifiers containing digits, such as the alias `F1`, are not changed.
2. **Vectorize.** Each distinct pattern becomes a TF-IDF vector. The tokenizer
   is the same as in `spv-sql.py`: words like `o.order_id` stay as one token.
   One-character words are dropped because they are almost always table
   aliases (`A`, `P`, `L`) and carry no meaning. Two-character aliases such as
   `CL` are still counted.
3. **Group.** Agglomerative clustering with average linkage on cosine
   distance. Groups keep merging until the average distance between two
   groups would exceed `--threshold`.
4. **Report.** Groups are listed in order of their share of executions.

### --no-keywords

SQL keywords and common functions (`select`, `from`, `where`, `and`, `count`,
`nvl`, `to_char`, ...) are removed before vectorizing, so similarity comes
from the tables and columns referenced.

- Double-quoted identifiers are never removed. Columns named after keywords,
  such as `"Date"`, `"Type"`, `"User"` or `"From"`, are kept.
- Column names still count. This is "tables and columns", not "tables only".
  Grouping on table names alone would require extracting them from the
  `FROM` and `JOIN` clauses.

## Cosine Similarity and the Threshold

### What cosine similarity measures

- **1.0**: the two statements use the same words in the same proportions.
  Word order is ignored, so `a JOIN b` and `b JOIN a` score 1.0.
- **0.0**: no words in common.
- TF-IDF weights are never negative, so the value is always between 0 and 1.
- Rare words carry more weight than common ones. Sharing an uncommon table
  such as `LANGCD` raises similarity far more than sharing `WHERE`.
- The script works with distance, which is `1 - similarity`.
  `--threshold 0.3` means "at least 0.7 similar".

With average linkage, the threshold applies to the average distance between
the members of two groups, not to every pair. A group formed at 0.7 can
contain pairs that are less than 0.7 similar to each other.

### Distribution in the sample workload

Similarity of each of the 962 distinct patterns to its closest other pattern
(default settings):

```
  0.00-0.30:   89 ##########
  0.30-0.50:  123 ###############
  0.50-0.70:  180 ######################
  0.70-0.90:  283 ###################################
  0.90-1.00:  287 ####################################
```

- 570 patterns have a close relative (0.7 or higher): variants of the same
  query with different column lists or extra predicates.
- About 90 patterns have nothing within 0.3 of them.
- Across all pairs the median similarity is only 0.026: the workload contains
  many unrelated query types.

### Effect of the threshold

Sample workload, default settings:

| `--threshold` | Min similarity | Groups | Single-pattern groups | Largest group, % of executions | Top 10 groups, % of executions |
|---|---|---|---|---|---|
| 0.05 | 0.95 | 855 | 780 | 5.3 | 23.1 |
| 0.10 | 0.90 | 793 | 680 | 5.3 | 23.8 |
| 0.20 | 0.80 | 697 | 539 | 5.3 | 24.6 |
| **0.30** (default) | 0.70 | 611 | 435 | 5.3 | 25.1 |
| 0.40 | 0.60 | 517 | 338 | 5.3 | 27.5 |
| 0.50 | 0.50 | 440 | 261 | 5.3 | 29.0 |
| 0.60 | 0.40 | 352 | 188 | 5.3 | 31.8 |
| 0.70 | 0.30 | 279 | 131 | 6.6 | 34.4 |
| 0.80 | 0.20 | 172 | 54 | 9.7 | 44.4 |

With `--no-keywords`:

| `--threshold` | Min similarity | Groups | Single-pattern groups | Largest group, % of executions | Top 10 groups, % of executions |
|---|---|---|---|---|---|
| 0.05 | 0.95 | 823 | 724 | 5.3 | 23.2 |
| 0.10 | 0.90 | 763 | 635 | 5.3 | 24.1 |
| 0.20 | 0.80 | 667 | 499 | 5.3 | 24.2 |
| **0.30** (default) | 0.70 | 580 | 394 | 5.3 | 25.1 |
| 0.40 | 0.60 | 503 | 323 | 5.3 | 25.6 |
| 0.50 | 0.50 | 421 | 253 | 5.3 | 28.9 |
| 0.60 | 0.40 | 355 | 198 | 6.5 | 33.0 |
| 0.70 | 0.30 | 294 | 144 | 6.5 | 33.7 |
| 0.80 | 0.20 | 203 | 76 | 9.4 | 39.5 |

Observations:

- The number of groups falls steadily as the threshold rises. There is no
  plateau marking a natural number of query families, so the threshold is a
  choice of how coarse a picture you want.
- The largest group stays at 5.3% until about 0.6. The `LANGCD` lookup is
  distinct enough that nothing joins it until similarity drops to about 0.3.
- Above about 0.7, unrelated queries begin to merge: at 0.2 similarity two
  statements may have little in common beyond `PATIENT` and a few column
  names. Groups become large but harder to interpret.
- `--no-keywords` follows nearly the same curve. Its main effect is on
  statements that only looked alike because of shared keywords: with
  keywords removed, 17 patterns have no neighbor above 0.1 similarity,
  compared with 2 when keywords are kept.

A reasonable starting point for workload characterization is
`--threshold 0.5`: about half as many single-pattern groups as the default,
with groups that are still recognizable. Check the example statements to
confirm the groups make sense for your workload.

## Scaling

The clustering step builds a full distance matrix, so memory and time grow
with the square of the number of **distinct normalized patterns**, not the
number of input lines. Because literals and bind variables are removed first,
a large trace file from an application that reuses its SQL still yields a
small number of patterns (5691 statements became 962 here). Tens of
thousands of truly distinct patterns, for example heavily generated ad hoc
SQL, would need a different clustering method.
