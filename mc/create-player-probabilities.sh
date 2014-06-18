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
  playerid,
  section_x,
  section_y,
  ((SUM(direction_x_0))/(SUM(total_x))) AS move_x_left,
  ((SUM(direction_x_1))/(SUM(total_x))) AS move_x_stay,
  ((SUM(direction_x_2))/(SUM(total_x))) AS move_x_right,
  ((SUM(direction_y_0))/(SUM(total_y))) AS move_y_up,
  ((SUM(direction_y_1))/(SUM(total_y))) AS move_y_stay,
  ((SUM(direction_y_2))/(SUM(total_y))) AS move_y_down
FROM [$MOVE_SUMMARY]
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
MOVE_PROBABILITY=$DATASET.move_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(shots_ontarget)/(SUM(shot_attempts))) AS shot_ontarget
FROM [$GAME_SUMMARY] WHERE
  shot_attempts > 0
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
SHOT_PROBABILITY=$DATASET.shot_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(pass_0)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_0,
  (SUM(pass_1)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_1,
  (SUM(pass_2)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_2,
  (SUM(pass_3)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_3,
  (SUM(pass_4)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_4,
  (SUM(pass_5)/(SUM(pass_attempt_0) + SUM(pass_attempt_1) + SUM(pass_attempt_2) + SUM(pass_attempt_3) + SUM(pass_attempt_4) + SUM(pass_attempt_5))) pass_5,
FROM [$GAME_SUMMARY] WHERE
  (pass_attempt_0 + pass_attempt_1 + pass_attempt_2 + pass_attempt_3 + pass_attempt_4 + pass_attempt_5) > 0
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
PASS_PROBABILITY=$DATASET.pass_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(save)/(SUM(save_attempts))) AS save
FROM [$GAME_SUMMARY] WHERE
  save_attempts > 0
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
SAVE_PROBABILITY=$DATASET.save_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(intercepts)/(SUM(intercept_attempts))) AS intercept
FROM [$GAME_SUMMARY] WHERE
  intercept_attempts > 0
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
INTERCEPT_PROBABILITY=$DATASET.intercept_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(tackles)/(SUM(tackle_attempts))) AS tackle
FROM [$GAME_SUMMARY] WHERE
  tackle_attempts > 0
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`
TACKLE_PROBABILITY=$DATASET.tackle_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(red_cards))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS red_card,
  (SUM(yellow_cards))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS yellow_card,
  (SUM(throwins_ontarget))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS throwin,
  (SUM(throwin_attempts))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS throwin_attempt,
  (SUM(free_kicks_ontarget))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS free_kick,
  (SUM(free_kick_attempts))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS free_kick_attempt,
  (SUM(penalty_kicks_ontarget))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS penalty_kick,
  (SUM(penalty_kick_attempts))/(SUM(throwin_attempts)+SUM(free_kick_attempts)+SUM(penalty_kick_attempts)+SUM(yellow_cards)+SUM(red_cards)) AS penalty_kick_attempt,
FROM [$GAME_SUMMARY] WHERE
  (throwin_attempts+free_kick_attempts+penalty_kick_attempts+yellow_cards+red_cards) > 0
  GROUP BY playerid, section_x, section_y
EOF`
FOUL_PROBABILITY=$DATASET.foul_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(throwins_ontarget))/(SUM(throwin_attempts)+SUM(corner_kick_attempts)) AS throwin,
  (SUM(throwin_attempts))/(SUM(throwin_attempts)+SUM(corner_kick_attempts)) AS throwin_attempt,
  (SUM(corner_kicks_ontarget))/(SUM(throwin_attempts)+SUM(corner_kick_attempts)) AS corner_kick,
  (SUM(corner_kick_attempts))/(SUM(throwin_attempts)+SUM(corner_kick_attempts)) AS corner_kick_attempt,
FROM [$GAME_SUMMARY] WHERE
  (throwin_attempts+corner_kick_attempts) > 0
  GROUP BY playerid, section_x, section_y
EOF`
OUT_PROBABILITY=$DATASET.out_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(tackles))/SUM(defense_event_count) AS tackle,
  (SUM(intercepts))/SUM(defense_event_count) AS intercept,
  (SUM(save))/SUM(defense_event_count) AS save
FROM [soccer.game_summary_0]
  GROUP BY playerid, section_x, section_y
EOF`
DEFENSE_PROBABILITY=$DATASET.defense_probabilities_player
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
  playerid,
  section_x,
  section_y,
  (SUM(tackles))/SUM(defense_event_count) AS tackle,
  (SUM(intercepts))/SUM(defense_event_count) AS intercept,
  (SUM(save))/SUM(defense_event_count) AS save
FROM [soccer.game_summary_0]
  GROUP BY playerid, section_x, section_y
EOF`
OFFSENSE_PROBABILITY=$DATASET.offsense_probabilities_player
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


