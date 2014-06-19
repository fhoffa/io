#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen


# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET


# Query to generate a mapping of team/player/time -> position
ESTIMATE_SUMMARY_QUERY=`cat<<EOF
SELECT 
 lhs.timestamp AS timestamp,
 lhs.periodid AS periodid,
 STRING(INTEGER(lhs.matchid)*-1) AS matchid,
 rhs.teamid AS teamid,
 rhs.altid AS playerid,
 lhs.x AS x,
 lhs.y AS y,
 lhs.typeid AS typeid,
 lhs.outcomeid AS outcomeid,
 lhs.qualifiers.type AS qualifiers.type,
 lhs.qualifiers.value AS qualifiers.value,
FROM [toque.touches] AS lhs 
INNER JOIN [toque.worldcup_roster_summary] AS rhs
ON lhs.playerid=rhs.playerid
EOF`


ESTIMATE_SUMMARY=$DATASET.touches_approximated
ESTIMATE_SUMMARY_JOB=ESTIMATE_SUMMARY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $ESTIMATE_SUMMARY"
bq \
  --project_id=$PROJECTID \
  --job_id=$ESTIMATE_SUMMARY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $ESTIMATE_SUMMARY \
        "$ESTIMATE_SUMMARY_QUERY"
bq \
  --project_id=$PROJECTID \
    wait $ESTIMATE_SUMMARY_JOB


