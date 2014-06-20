#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=${DATASET:-$USER}


if [ -z "$1" ]; then
  echo 'Generate a set of probability inputs for a game given some teamid:"
  echo $0 ' [teamid] (e.g. 1216,1411,3273,...)'
  exit 1
fi
TEAMID=$1


MOVE_SUMMARY=$DATASET.move_summary
GAME_SUMMARY=$DATASET.game_summary
POSITION_SUMMARY=$DATASET.position_summary


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
  WHERE teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`

# Query to generate a mapping of player -> shot probability
SHOT_PROBABILITY_QUERY=`cat<<EOF
SELECT 
  playerid,
  section_x,
  section_y,
  (SUM(shots_ontarget)/(SUM(shot_attempts))) AS shot_ontarget
FROM [$GAME_SUMMARY] WHERE
  shot_attempts > 0 AND
  teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`


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
  (pass_attempt_0 + pass_attempt_1 + pass_attempt_2 + pass_attempt_3 + pass_attempt_4 + pass_attempt_5) > 0 AND
  teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`


# Query to generate a mapping of player -> save probability
SAVE_PROBABILITY_QUERY=`cat<<EOF
SELECT 
  playerid,
  section_x,
  section_y,
  (SUM(save)/(SUM(save_attempts))) AS save
FROM [$GAME_SUMMARY] WHERE
  save_attempts > 0 AND
  teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`


# Query to generate a mapping of player -> intercept probability
INTERCEPT_PROBABILITY_QUERY=`cat<<EOF
SELECT 
  playerid,
  section_x,
  section_y,
  (SUM(intercepts)/(SUM(intercept_attempts))) AS intercept
FROM [$GAME_SUMMARY] WHERE
  intercept_attempts > 0 AND
  teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`


# Query to generate a mapping of player -> tackle probability
TACKLE_PROBABILITY_QUERY=`cat<<EOF
SELECT 
  playerid,
  section_x,
  section_y,
  (SUM(tackles)/(SUM(tackle_attempts))) AS tackle
FROM [$GAME_SUMMARY] WHERE
  tackle_attempts > 0 AND
  teamid=$TEAMID
GROUP BY
  playerid,
  section_x,
  section_y
ORDER BY
  playerid,
  section_x
EOF`


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
  (throwin_attempts+free_kick_attempts+penalty_kick_attempts+yellow_cards+red_cards) > 0 AND
  teamid=$TEAMID
  GROUP BY playerid, section_x, section_y
EOF`


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
  (throwin_attempts+corner_kick_attempts) > 0 AND
  teamid=$TEAMID
  GROUP BY playerid, section_x, section_y
EOF`


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
WHERE teamid=$TEAMID
  GROUP BY playerid, section_x, section_y
EOF`


# Query to generate a mapping of player -> offense probability
OFFENSE_PROBABILITY_QUERY=`cat<<EOF
SELECT
  playerid,
  section_x,
  section_y,
  (SUM(tackles))/SUM(defense_event_count) AS tackle,
  (SUM(intercepts))/SUM(defense_event_count) AS intercept,
  (SUM(save))/SUM(defense_event_count) AS save
FROM [soccer.game_summary_0]
WHERE teamid=$TEAMID
  GROUP BY playerid, section_x, section_y
EOF`


# Generate a JSON input for the simulation the describes all of the aspects 
# of each player.
cat<<EOF
{
    "move": `query_to_json "$MOVE_PROBABILITY_QUERY"`,
    "shot":`query_to_json "$SHOT_PROBABILITY_QUERY"`,
    "save": `query_to_json "$SAVE_PROBABILITY_QUERY"`,
    "pass": `query_to_json "$PASS_PROBABILITY_QUERY"`,
    "intercept": `query_to_json "$INTERCEPT_PROBABILITY_QUERY"`,
    "tackle": `query_to_json "$TACKLE_PROBABILITY_QUERY"`,
    "foul": `query_to_json "$FOUL_PROBABILITY_QUERY"`,
    "out": `query_to_json "$OUT_PROBABILITY_QUERY"`,
    "defense": `query_to_json "$DEFENSE_PROBABILITY_QUERY"`,
    "offense": `query_to_json "$OFFENSE_PROBABILITY_QUERY"`
}
EOF

