#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen

if [ -z "$1" -o -z "$2" ]; then
  echo 'Generate a game_state input starting from some point in a given match:'
  echo $0 ' [matchid] [timestamp-between-0-and-1]'
  exit 1
fi
MATCHID=$1
TIMESTAMP=$2

GAME_STATE=$DATASET.game_state

# Generate a temporary table w/ rolled up stats but no locations
GAME_STATE_TMP_QUERY=`cat<<EOF
SELECT 
 lhs.timestamp AS timestamp,
 lhs.matchid AS matchid, 
 lhs.teamid AS teamid,
 lhs.position AS position,
 lhs.playerid AS playerid,
 lhs.fouls AS fouls,
 lhs.outs AS outs,
 lhs.red_cards AS red_cards,
 lhs.yellow_cards AS yellow_cards,
 lhs.pass AS pass,
 lhs.pass_attempts AS pass_attempts,
 lhs.save AS save,
 lhs.save_attempts AS save_attempts,
 lhs.corner_kicks_ontarget AS corner_kicks_ontarget,
 lhs.corner_kick_attempts AS corner_kick_attempts,
 lhs.penalty_kicks_ontarget AS penalty_kicks_ontarget,
 lhs.penalty_kick_attempts AS penalty_kick_attempts,
 lhs.free_kicks_ontarget AS free_kicks_ontarget,
 lhs.free_kick_attempts AS free_kick_attempts,
 lhs.throwins_ontarget AS throwins_ontarget,
 lhs.throwin_attempts AS throwin_attempts,
 lhs.shots_ontarget AS shots_ontarget,
 lhs.shot_attempts AS shot_attempts,
 lhs.goals AS goals,
 lhs.own_goals AS own_goals,
 lhs.intercepts AS intercepts,
 lhs.intercept_attempts AS intercept_attempts,
 lhs.tackles AS tackles,
 lhs.tackle_attempts AS tackle_attempts
  FROM [$GAME_STATE] AS lhs
LEFT JOIN (SELECT
    playerid, teamid, matchid, MAX(timestamp) AS timestamp,
   FROM [$GAME_STATE]
WHERE
  matchid='731767' AND timestamp<=0.25
GROUP BY
  playerid, teamid, matchid) AS rhs ON
lhs.playerid=rhs.playerid AND lhs.teamid=rhs.teamid AND lhs.matchid=rhs.matchid
EOF`
GAME_STATE_TMP=$DATASET.game_state_tmp_$$
GAME_STATE_TMP_JOB=GAME_STATE_TMP_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $GAME_STATE"
bq \
  --project_id=$PROJECTID \
  --job_id=$GAME_STATE_TMP_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $GAME_STATE_TMP \
      --quiet \
        "$GAME_STATE_TMP_QUERY"
bq \
  --project_id=$PROJECTID \
    wait $GAME_STATE_TMP_JOB


# Add the most recent player location to the rolled up stats
GAME_STATE_QUERY=`cat<<EOF
SELECT 
 lhs.matchid AS matchid, 
 lhs.x AS x,
 lhs.y AS y,
 lhs.timestamp AS timestamp,
 lhs.teamid AS teamid,
 lhs.position AS position,
 lhs.playerid AS playerid,
 lhs.fouls AS fouls,
 lhs.outs AS outs,
 lhs.red_cards AS red_cards,
 lhs.yellow_cards AS yellow_cards,
 lhs.pass AS pass,
 lhs.pass_attempts AS pass_attempts,
 lhs.save AS save,
 lhs.save_attempts AS save_attempts,
 lhs.corner_kicks_ontarget AS corner_kicks_ontarget,
 lhs.corner_kick_attempts AS corner_kick_attempts,
 lhs.penalty_kicks_ontarget AS penalty_kicks_ontarget,
 lhs.penalty_kick_attempts AS penalty_kick_attempts,
 lhs.free_kicks_ontarget AS free_kicks_ontarget,
 lhs.free_kick_attempts AS free_kick_attempts,
 lhs.throwins_ontarget AS throwins_ontarget,
 lhs.throwin_attempts AS throwin_attempts,
 lhs.shots_ontarget AS shots_ontarget,
 lhs.shot_attempts AS shot_attempts,
 lhs.goals AS goals,
 lhs.own_goals AS own_goals,
 lhs.intercepts AS intercepts,
 lhs.intercept_attempts AS intercept_attempts,
 lhs.tackles AS tackles,
 lhs.tackle_attempts AS tackle_attempts
FROM [$GAME_STATE] AS lhs
LEFT OUTER JOIN EACH [$GAME_STATE_TMP] AS rhs
ON lhs.playerid=rhs.playerid AND lhs.matchid=rhs.matchid AND lhs.timestamp=rhs.timestamp AND lhs.teamid=rhs.teamid

EOF`
GAME_STATE_JOB=GAME_STATE_$SECONDS_$RANDOM_$RANDOM_$$
bq \
  --project_id=$PROJECTID \
  --job_id=$GAME_STATE_JOB \
    query \
      --format=json \
      --quiet \
      "$GAME_STATE_QUERY"

bq \
  --project_id=$PROJECTID \
  rm -f \
    "$GAME_STATE_TMP"
