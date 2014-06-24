#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=${DATASET:-$USER}

if [ -z "$1" -o -z "$2" ]; then
  echo 'Generate a game_state input starting from some point in a given match:'
  echo $0 ' [matchid] [timestamp-between-0-and-1]'
  exit 1
fi
MATCHID=$1
TIMESTAMP=$2


function query_to_json() {
 X=$(echo `bq \
      --project_id=$PROJECTID \
      --job_id=$MOVE_PROBABILITY_JOB \
        query \
          --quiet \
          --format=json \
            $1 | sed -e s:^\"::g -e s:\"\$::g -e s:\':\\\\\":g -e s:\':\":g`)
 [ x"$X" == x ] && X='[]'
 echo $X
}


# Add the most recent player location to the rolled up stats
TOUCHES_SUMMARY=$DATASET.touches_summary
STATE_SUMMARY=$DATASET.state_summary
STATE_SUMMARY_QUERY=`cat<<EOF
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
FROM [$TOUCHES_SUMMARY] AS lhs
LEFT OUTER JOIN EACH (SELECT 
 innerlhs.timestamp AS timestamp,
 innerlhs.matchid AS matchid, 
 innerlhs.teamid AS teamid,
 innerlhs.position AS position,
 innerlhs.playerid AS playerid,
 innerlhs.fouls AS fouls,
 innerlhs.outs AS outs,
 innerlhs.red_cards AS red_cards,
 innerlhs.yellow_cards AS yellow_cards,
 innerlhs.pass AS pass,
 innerlhs.pass_attempts AS pass_attempts,
 innerlhs.save AS save,
 innerlhs.save_attempts AS save_attempts,
 innerlhs.corner_kicks_ontarget AS corner_kicks_ontarget,
 innerlhs.corner_kick_attempts AS corner_kick_attempts,
 innerlhs.penalty_kicks_ontarget AS penalty_kicks_ontarget,
 innerlhs.penalty_kick_attempts AS penalty_kick_attempts,
 innerlhs.free_kicks_ontarget AS free_kicks_ontarget,
 innerlhs.free_kick_attempts AS free_kick_attempts,
 innerlhs.throwins_ontarget AS throwins_ontarget,
 innerlhs.throwin_attempts AS throwin_attempts,
 innerlhs.shots_ontarget AS shots_ontarget,
 innerlhs.shot_attempts AS shot_attempts,
 innerlhs.goals AS goals,
 innerlhs.own_goals AS own_goals,
 innerlhs.intercepts AS intercepts,
 innerlhs.intercept_attempts AS intercept_attempts,
 innerlhs.tackles AS tackles,
 innerlhs.tackle_attempts AS tackle_attempts
  FROM [$TOUCHES_SUMMARY] AS innerlhs
LEFT JOIN (SELECT
    playerid, teamid, matchid, MAX(timestamp) AS timestamp,
   FROM [$TOUCHES_SUMMARY]
WHERE
  matchid=$MATCHID AND timestamp<=$TIMESTAMP
GROUP BY
  playerid, teamid, matchid) AS innerrhs ON
innerlhs.playerid=innerrhs.playerid AND innerlhs.teamid=innerrhs.teamid AND innerlhs.matchid=innerrhs.matchid) AS rhs
ON lhs.playerid=rhs.playerid AND lhs.matchid=rhs.matchid AND lhs.timestamp=rhs.timestamp AND lhs.teamid=rhs.teamid
ORDER BY lhs.teamid, lhs.timestamp
EOF`


cat<<EOF
{
    "state": `query_to_json "$STATE_SUMMARY_QUERY"`
}
EOF
