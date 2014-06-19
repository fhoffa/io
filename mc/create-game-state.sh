#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen

GAME_SUMMARY=$DATASET.game_summary


# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET


GAME_STATE_QUERY=`cat<<EOF
SELECT
 ((lhs.second-rhs.min)/(rhs.max-rhs.min)) AS timestamp,
 lhs.matchid AS matchid, 
 lhs.teamid AS teamid,
 lhs.position AS position,
 lhs.playerid AS playerid,
 lhs.x AS x,
 lhs.y AS y,
 lhs.cum_fouls AS fouls,
 lhs.cum_outs AS outs,
 lhs.cum_red_cards AS red_cards,
 lhs.cum_yellow_cards AS yellow_cards,
 (lhs.cum_pass_0+lhs.cum_pass_1+lhs.cum_pass_2+lhs.cum_pass_3+lhs.cum_pass_4+lhs.cum_pass_5) AS pass,
 (lhs.cum_pass_attempt_0+lhs.cum_pass_attempt_1+lhs.cum_pass_attempt_2+lhs.cum_pass_attempt_3+lhs.cum_pass_attempt_4+lhs.cum_pass_attempt_5) AS pass_attempts,
 lhs.cum_save AS save,
 lhs.cum_save_attempts AS save_attempts,
 lhs.cum_corner_kicks_ontarget AS corner_kicks_ontarget,
 lhs.cum_corner_kick_attempts AS corner_kick_attempts,
 lhs.cum_penalty_kicks_ontarget AS penalty_kicks_ontarget,
 lhs.cum_penalty_kick_attempts AS penalty_kick_attempts,
 lhs.cum_free_kicks_ontarget AS free_kicks_ontarget,
 lhs.cum_free_kick_attempts AS free_kick_attempts,
 lhs.cum_throwins_ontarget AS throwins_ontarget,
 lhs.cum_throwin_attempts AS throwin_attempts,
 lhs.cum_shots_ontarget AS shots_ontarget,
 lhs.cum_shot_attempts AS shot_attempts,
 lhs.cum_goals AS goals,
 lhs.cum_own_goals AS own_goals,
 lhs.cum_intercepts AS intercepts,
 lhs.cum_intercept_attempts AS intercept_attempts,
 lhs.cum_tackles AS tackles,
 lhs.cum_tackle_attempts AS tackle_attempts
 FROM (SELECT 
    matchid, second, periodid, position, playerid, teamid, x, y,
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
    sum(goals) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_goals,
    sum(own_goals) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_own_goals,
    sum(intercepts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_intercepts,
    sum(intercept_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_intercept_attempts,
    sum(tackles) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_tackles,
    sum(tackle_attempts) OVER (PARTITION BY playerid, matchid ORDER BY second) as cum_tackle_attempts,
  FROM [$GAME_SUMMARY]) AS lhs
    JOIN (SELECT matchid, MIN(second) AS min, MAX(second) AS max FROM [$GAME_SUMMARY] GROUP BY matchid) AS rhs
      ON lhs.matchid = rhs.matchid
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


