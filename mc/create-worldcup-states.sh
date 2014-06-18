#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen

GAME_SUMMARY=$DATASET.game_summary


# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET


GAME_STATE_QUERY="SELECT * FROM [$GAME_SUMMARY] WHERE matchid='731767'"
GAME_STATE=$DATASET.worldcup_game_state
GAME_STATE_JOB=GAME_STATE_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $GAME_STATE"
bq \
  --project_id=$PROJECTID \
  --job_id=$GAME_STATE_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $GAME_STATE \
        "$GAME_STATE_QUERY"
bq \
  --project_id=$PROJECTID \
    wait $GAME_STATE_JOB

