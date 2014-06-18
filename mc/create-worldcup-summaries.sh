#!/bin/bash -ex
PROJECTID=cloude-sandbox
DATASET=crahen

GAME_SUMMARY=$DATASET.game_summary

# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET

# Query to generate a mapping of player -> move probability
WORDCUP_GAME_SUMMARY_QUERY="SELECT * FROM [$GAME_SUMMARY] WHERE matchid='731767'"
WORDCUP_GAME_SUMMARY=$DATASET.worldcup_game_summary
WORDCUP_GAME_SUMMARY_JOB=WORDCUP_GAME_SUMMARY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $WORDCUP_GAME_SUMMARY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$WORDCUP_GAME_SUMMARY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $WORDCUP_GAME_SUMMARY \
        "$WORDCUP_GAME_SUMMARY_QUERY"

bq \
  --project_id=$PROJECTID \
    wait $WORDCUP_GAME_SUMMARY_JOB
