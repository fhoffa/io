#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen

GAME_SUMMARY=$DATASET.game_summary


# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET


GAME_STATE_QUERY=`cat<<EOF
SELECT 
  matchid, second, periodid, playerid, teamid, section_x, section_y, x, y,
  sum(fouls) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_fouls,
  sum(outs) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_outs,
  sum(red_cards) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_red_cards,
  sum(yellow_cards) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_yellow_cards,
  sum(pass_0) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_0,
  sum(pass_1) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_1,
  sum(pass_2) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_2,
  sum(pass_3) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_3,
  sum(pass_4) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_4,
  sum(pass_5) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_5,
  sum(pass_attempt_0) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_0,
  sum(pass_attempt_1) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_1,
  sum(pass_attempt_2) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_2,
  sum(pass_attempt_3) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_3,
  sum(pass_attempt_4) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_4,
  sum(pass_attempt_5) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_pass_attempt_5,
  sum(save) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_save,
  sum(save_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_save_attempts, 
  sum(corner_kicks_ontarget) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_corner_kicks_ontarget,
  sum(corner_kick_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_corner_kick_attempts,
  sum(penalty_kicks_ontarget) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_penalty_kicks_ontarget,
  sum(penalty_kick_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_penalty_kick_attempts,
  sum(free_kicks_ontarget) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_free_kicks_ontarget,
  sum(free_kick_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_free_kick_attempts, 
  sum(throwins_ontarget) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_throwins_ontarget,
  sum(throwin_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_throwin_attempts,
  sum(shots_ontarget) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_shots_ontarget,
  sum(shot_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_shot_attempts,
  sum(own_goals) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_own_goals,
  sum(intercepts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_intercepts,
  sum(intercept_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_intercept_attempts,
  sum(tackles) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_tackles,
  sum(tackle_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_tackle_attempts,
FROM [$GAME_SUMMARY]
EOF`
GAME_STATE=$DATASET.game_state
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

