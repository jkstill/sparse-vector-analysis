
## SQL Text Acquisition

Ideally a text file containing the number of executions and the SQL text can be used to drive the analysis. The file should be structured as follows:

I am using Method R tools to extract this information from Oracle SQL (10046) trace files.

```bash
mrskew --nohead --nofoot --nodashes --nocommas \
    --sep=$'\t' --top=0 --name='^EXEC$' \
    --group='$sql' --gl='SQL_TEXT' \
    --select='1' --sl='EXECS' --sort=2nd 
    --where='$dep==0' *.trc \
    | perl -F'\t' -lane '$F[0]=~s/\s+$//; printf "%d\t%s\n", $F[1], $F[0]' \
    > sql_text.txt


