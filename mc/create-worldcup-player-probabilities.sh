#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen

MOVE_SUMMARY=$DATASET.move_summary
GAME_SUMMARY=$DATASET.game_summary
POSITION_SUMMARY=$DATASET.position_summary

# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET

# Query to generate a mapping of player -> move probability
MOVE_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.move_x_left AS move_x_left,
    lhs.move_x_stay AS move_x_stay,
    lhs.move_x_right AS move_x_right,
    lhs.move_y_down AS move_y_down,
    lhs.move_y_stay AS move_y_stay,
    lhs.move_y_up AS move_y_up,
  FROM [soccer.move_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
MOVE_PROBABILITY=$DATASET.worldcup_move_probabilities_player
MOVE_PROBABILITY_JOB=MOVE_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $MOVE_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$MOVE_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $MOVE_PROBABILITY \
        "$MOVE_PROBABILITY_QUERY"

# Query to generate a mapping of player -> shot probability
SHOT_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.shot_ontarget AS shot_ontarget,
  FROM [soccer.shot_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
SHOT_PROBABILITY=$DATASET.worldcup_shot_probabilities_player
SHOT_PROBABILITY_JOB=SHOT_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $SHOT_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$SHOT_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $SHOT_PROBABILITY \
        "$SHOT_PROBABILITY_QUERY"


# Query to generate a mapping of player -> pass probability
PASS_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.pass_0 AS pass_0,
    lhs.pass_1 AS pass_1,
    lhs.pass_2 AS pass_2,
    lhs.pass_3 AS pass_3,
    lhs.pass_4 AS pass_4,
    lhs.pass_5 AS pass_5,
  FROM [soccer.pass_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
PASS_PROBABILITY=$DATASET.worldcup_pass_probabilities_player
PASS_PROBABILITY_JOB=PASS_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $PASS_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$PASS_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $PASS_PROBABILITY \
        "$PASS_PROBABILITY_QUERY"


# Query to generate a mapping of player -> save probability
SAVE_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.save AS saves,
  FROM [soccer.save_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
SAVE_PROBABILITY=$DATASET.worldcup_save_probabilities_player
SAVE_PROBABILITY_JOB=SAVE_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $SAVE_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$SAVE_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $SAVE_PROBABILITY \
        "$SAVE_PROBABILITY_QUERY"


# Query to generate a mapping of player -> intercept probability
INTERCEPT_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.intercept AS intercepts,
  FROM [soccer.intercept_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
INTERCEPT_PROBABILITY=$DATASET.worldcup_intercept_probabilities_player
INTERCEPT_PROBABILITY_JOB=INTERCEPT_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $INTERCEPT_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$INTERCEPT_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $INTERCEPT_PROBABILITY \
        "$INTERCEPT_PROBABILITY_QUERY"



# Query to generate a mapping of player -> tackle probability
TACKLE_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.tackle AS tackles,
  FROM [soccer.tackle_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
TACKLE_PROBABILITY=$DATASET.worldcup_tackle_probabilities_player
TACKLE_PROBABILITY_JOB=TACKLE_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $TACKLE_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$TACKLE_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $TACKLE_PROBABILITY \
        "$TACKLE_PROBABILITY_QUERY"


# Query to generate a mapping of player -> foul probability
FOUL_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.red_card AS red_card,
    lhs.yellow_card AS yellow_card,
    lhs.throwin AS throwin,
    lhs.throwin_attempt AS throwin_attempt,
    lhs.free_kick AS free_kick,
    lhs.free_kick_attempt AS free_kick_attempt,
    lhs.penalty_kick AS penalty_kick,
    lhs.penalty_kick_attempt AS penalty_kick_attempt,
  FROM [soccer.foul_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
FOUL_PROBABILITY=$DATASET.worldcup_foul_probabilities_player
FOUL_PROBABILITY_JOB=FOUL_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $FOUL_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$FOUL_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $FOUL_PROBABILITY \
        "$FOUL_PROBABILITY_QUERY"


# Query to generate a mapping of player -> out probability
OUT_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.throwin AS throwin,
    lhs.throwin_attempt AS throwin_attempt,
    lhs.corner_kick AS corner_kick,
    lhs.corner_kick_attempt AS corner_kick_attempt,
  FROM [soccer.out_probabilities_player] AS lhs  
    LEFT OUTER JOIN [toque.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
OUT_PROBABILITY=$DATASET.worldcup_out_probabilities_player
OUT_PROBABILITY_JOB=OUT_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $OUT_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$OUT_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $OUT_PROBABILITY \
        "$OUT_PROBABILITY_QUERY"


bq \
  --project_id=$PROJECTID \
    wait $MOVE_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $SHOT_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $PASS_PROBABILITY_JOB


bq \
  --project_id=$PROJECTID \
    wait $SAVE_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $INTERCEPT_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $TACKLE_PROBABILITY_JOB

bq \
  --project_id=$PROJECTID \
    wait $FOUL_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $OUT_PROBABILITY_JOB


# Query to generate a mapping of player -> defense probability
DEFENSE_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.position AS position,
    lhs.tackle AS tackle,
    lhs.intercept AS intercept,
    lhs.save AS save,
  FROM [soccer.defense_probabilities_player] AS lhs  
    JOIN [soccer.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
DEFENSE_PROBABILITY=$DATASET.worldcup_defense_probabilities_player
DEFENSE_PROBABILITY_JOB=DEFENSE_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $DEFENSE_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$DEFENSE_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $DEFENSE_PROBABILITY \
        "$DEFENSE_PROBABILITY_QUERY"


# Query to generate a mapping of player -> offense probability
OFFSENSE_PROBABILITY_QUERY=`cat<<EOF
SELECT
    lhs.playerid AS playerid,
    lhs.section_x AS section_x,
    lhs.section_y AS section_y,
    lhs.throwin AS throwin,
    lhs.throwin_attempt AS throwin_attempt,
    lhs.corner_kick AS corner_kick,
    lhs.corner_kick_attempt AS corner_kick_attempt,
  FROM [soccer.out_probabilities_player] AS lhs  
    LEFT OUTER JOIN [soccer.worldcup_roster_summary] AS rhs
      ON lhs.playerid == rhs.altid
EOF`
OFFSENSE_PROBABILITY=$DATASET.worldcup_offsense_probabilities_player
OFFSENSE_PROBABILITY_JOB=OFFSENSE_PROBABILITY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $OFFSENSE_PROBABILITY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$OFFSENSE_PROBABILITY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $OFFSENSE_PROBABILITY \
        "$OFFSENSE_PROBABILITY_QUERY"


bq \
  --project_id=$PROJECTID \
    wait $OFFSENSE_PROBABILITY_JOB
bq \
  --project_id=$PROJECTID \
    wait $DEFENSE_PROBABILITY_JOB


