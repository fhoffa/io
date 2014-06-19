#!/bin/bash -e
PROJECTID=cloude-sandbox
DATASET=crahen


# Create the dataset
bq --project_id=$PROJECTID \
  mk -f -d $DATASET


# Query to generate a mapping of team/player/time -> position
POSITION_SUMMARY_QUERY=`cat<<EOF
SELECT matchid, INTEGER(FLOOR(TIMESTAMP_TO_MSEC(timestamp)/1000)) AS second, periodid, teamid, playerid, LOWER(qualifiers.value) AS position 
  FROM [toque.touches] 
    WHERE qualifiers.type == 44 AND playerid != 0
      ORDER BY matchid, second, periodid
EOF`

POSITION_SUMMARY=$DATASET.position_summary
POSITION_SUMMARY_JOB=POSITION_SUMMARY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $POSITION_SUMMARY"
bq \
  --project_id=$PROJECTID \
  --job_id=$POSITION_SUMMARY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $POSITION_SUMMARY \
        "$POSITION_SUMMARY_QUERY"
bq \
  --project_id=$PROJECTID \
    wait $POSITION_SUMMARY_JOB


# Query to generate a second-by-second play summay of the game we can build statistics from.
GAME_SUMMARY_QUERY=`cat<<EOF
SELECT matchid, second, periodid, teamid, playerid, position, section_x, section_y, x, y,
  (SUM(foul) +
   SUM(out) +
   SUM(yellow_card) +
   SUM(red_card) +
   SUM(pass_attempt_0) +
   SUM(pass_attempt_1) +
   SUM(pass_attempt_2) +
   SUM(pass_attempt_3) +
   SUM(pass_attempt_4) +
   SUM(pass_attempt_5) +
   SUM(corner_kick_attempt) +
   SUM(penalty_kick_attempt) +
   SUM(free_kick_attempt) +
   SUM(throwin_attempt) +
   SUM(shot_attempt) +
   SUM(save_attempt) +
   SUM(intercept_attempt)) AS event_count,
  (SUM(foul) +
   SUM(out) +
   SUM(pass_attempt_0) +
   SUM(pass_attempt_1) +
   SUM(pass_attempt_2) +
   SUM(pass_attempt_3) +
   SUM(pass_attempt_4) +
   SUM(pass_attempt_5) +
   SUM(shot_attempt)) AS offense_event_count,
  (SUM(foul) +
   SUM(tackle_attempt) +
   SUM(intercept_attempt) +
   SUM(save_attempt)) AS defense_event_count,
  SUM(foul) AS fouls,
  SUM(out) AS outs,
  SUM(yellow_card) as yellow_cards,
  SUM(red_card) AS red_cards, 
  SUM(pass_0) AS pass_0,
  SUM(pass_1) AS pass_1,
  SUM(pass_2) AS pass_2,
  SUM(pass_3) AS pass_3,
  SUM(pass_4) AS pass_4,
  SUM(pass_5) AS pass_5,
  SUM(pass_attempt_0) AS pass_attempt_0, 
  SUM(pass_attempt_1) AS pass_attempt_1, 
  SUM(pass_attempt_2) AS pass_attempt_2, 
  SUM(pass_attempt_3) AS pass_attempt_3, 
  SUM(pass_attempt_4) AS pass_attempt_4, 
  SUM(pass_attempt_5) AS pass_attempt_5, 
  SUM(save) AS save,
  SUM(save_attempt) AS save_attempts,
  SUM(corner_kick_attempt) AS corner_kick_attempts,
  SUM(corner_kick_ontarget) AS corner_kicks_ontarget,
  SUM(penalty_kick_attempt) AS penalty_kick_attempts,
  SUM(penalty_kick_ontarget) AS penalty_kicks_ontarget,
  SUM(free_kick_attempt) AS free_kick_attempts,
  SUM(free_kick_ontarget) AS free_kicks_ontarget,
  SUM(throwin_attempt) AS throwin_attempts,
  SUM(throwin_ontarget) AS throwins_ontarget,
  SUM(shot_attempt) AS shot_attempts,
  SUM(shot_ontarget) AS shots_ontarget,
  SUM(goal) AS goals,
  SUM(own_goal) AS own_goals,
  SUM(intercept) AS intercepts,
  SUM(intercept_attempt) AS intercept_attempts,
  SUM(tackle) AS tackles,
  SUM(tackle_attempt) AS tackle_attempts
 FROM (
 SELECT INTEGER(FLOOR(TIMESTAMP_TO_MSEC(timestamp)/1000)) AS second, periodid, team_eventid, lhs.matchid AS matchid, lhs.teamid AS teamid, lhs.playerid AS playerid, rhs.position AS position, x, y,
   if (x <= 16.6, 0, if(x <= 33.2, 1, if(x <= 49.9, 2, if(x <= 66.4, 3, if(x <= 83, 4, 5))))) AS section_x,
   if (y <= 33, 0, if(y <= 66, 1, 2)) AS section_y, 
   if (typeid == 4 and outcomeid == 1, 1, 0) as foul,
   if (typeid == 5 and outcomeid == 1, 1, 0) as out,
   if (typeid == 17 AND qualifiers.type in (31), 1, 0) as yellow_card,
   if (typeid == 17 AND qualifiers.type in (32,33), 1, 0) as red_card,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) <= 16.6, 1, 0), 0) as pass_0,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) > 16.6 AND FLOAT(qualifiers.value) <= 33.2, 1, 0), 0) as pass_1,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) > 33.2 AND FLOAT(qualifiers.value) <= 49.9, 1, 0), 0) as pass_2,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) > 49.9 AND FLOAT(qualifiers.value) <= 66.4, 1, 0), 0) as pass_3,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) > 66.4 AND FLOAT(qualifiers.value) <= 83, 1, 0), 0) as pass_4,
   if (typeid == 1 AND qualifiers.type == 140 and outcomeid == 1, if (FLOAT(qualifiers.value) > 83, 1, 0), 0) as pass_5,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) <= 16.6, 1, 0), 0) as pass_attempt_0,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) > 16.6 AND FLOAT(qualifiers.value) <= 33.2, 1, 0), 0) as pass_attempt_1,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) > 33.2 AND FLOAT(qualifiers.value) <= 49.9, 1, 0), 0) as pass_attempt_2,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) > 49.9 AND FLOAT(qualifiers.value) <= 66.4, 1, 0), 0) as pass_attempt_3,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) > 66.4 AND FLOAT(qualifiers.value) <= 83, 1, 0), 0) as pass_attempt_4,
   if (typeid == 1 AND qualifiers.type == 140, if (FLOAT(qualifiers.value) > 83, 1, 0), 0) as pass_attempt_5,
   if (typeid == 10 AND qualifiers.type != 94 and outcomeid == 1, 1, 0) as save,
   if (typeid == 10 AND qualifiers.type != 94, 1, 0) as save_attempt,
   if (typeid in (13, 14, 15, 16) AND qualifiers.type in (25), 1, 0) as corner_kick_attempt,
   if (typeid in (15, 16) AND qualifiers.type in (25), 1, 0) as corner_kick_ontarget,
   if (typeid in (13, 14, 15, 16) AND qualifiers.type in (24, 16), 1, 0) as free_kick_attempt,
   if (typeid in (15, 16) AND qualifiers.type in (24, 16), 1, 0) as free_kick_ontarget,
   if (typeid in (13, 14, 15, 16) AND qualifiers.type in (160), 1, 0) as throwin_attempt,
   if (typeid in (15, 16) AND qualifiers.type in (160), 1, 0) as throwin_ontarget,
   if (typeid in (13, 14, 15, 16) AND qualifiers.type in (9), 1, 0) as penalty_kick_attempt,
   if (typeid in (15, 16) AND qualifiers.type in (9), 1, 0) as penalty_kick_ontarget,
   if (typeid in (13, 14) AND NOT qualifiers.type in (9, 24, 25, 26, 28, 160), 1, 0) as shot_attempt,
   if (typeid in (15, 16) AND NOT qualifiers.type in (9, 24, 25, 26, 28, 160), 1, 0) as shot_ontarget,
   if (typeid in (16) AND qualifiers.type not in (28) AND outcomeid == 1, 1, 0) as goal,  
   if (typeid in (16) AND qualifiers.type in (28) AND outcomeid == 1, 1, 0) as own_goal,  
   if (typeid in (8, 74) and outcomeid == 1 , 1, 0) as intercept,  
   if (typeid in (8, 74), 1, 0) as intercept_attempt,
   if (typeid in (7) and outcomeid == 1 , 1, 0) as tackle,  
   if (typeid in (7), 1, 0) as tackle_attempt,
   FROM [toque.touches] AS lhs
     JOIN (SELECT matchid, teamid, playerid, position FROM [$POSITION_SUMMARY]) AS rhs 
       ON lhs.matchid=rhs.matchid AND lhs.teamid=rhs.teamid AND lhs.playerid=rhs.playerid
         WHERE 
           (typeid in (4,5,7,8,74)) OR
           (typeid in (10) and qualifiers.type != 94) OR
           (typeid in (1) and qualifiers.type == 140) OR 
           (typeid in (13, 14, 15, 16) AND outcomeid == 1) OR
           (typeid == 17 AND qualifiers.type in (31, 32, 33))  
) WHERE
   (foul > 0 OR
    out > 0 OR
    goal > 0 OR
    own_goal > 0 OR
    shot_attempt > 0 OR
    shot_ontarget > 0 OR
    penalty_kick_ontarget > 0 OR
    penalty_kick_attempt > 0 OR
    throwin_ontarget > 0 OR
    throwin_attempt > 0 OR
    free_kick_ontarget > 0 OR
    free_kick_attempt > 0 OR
    corner_kick_ontarget > 0 OR
    corner_kick_attempt > 0 OR
    red_card > 0 OR
    intercept_attempt > 0 OR
    tackle_attempt > 0 OR
    save_attempt > 0 OR
    pass_0 > 0 OR pass_1 > 0 OR pass_2 > 0 OR pass_3 > 0 OR pass_4 > 0 OR pass_5 > 0 OR 
    pass_attempt_0 > 0 OR pass_attempt_1 > 0 OR pass_attempt_2 > 0 OR pass_attempt_3 > 0 OR pass_attempt_4 > 0 OR pass_attempt_5 > 0 OR 
    yellow_card > 0)
  GROUP EACH BY matchid, second, periodid, teamid, playerid, position, section_x, section_y, x ,y
  ORDER EACH BY matchid, second, periodid
EOF`

GAME_SUMMARY=$DATASET.game_summary
GAME_SUMMARY_JOB=GAME_SUMMARY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $GAME_SUMMARY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$GAME_SUMMARY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $GAME_SUMMARY \
        "$GAME_SUMMARY_QUERY"


# Query to generate a second-by-second movement summay of the game we can build statistics from.
MOVE_SUMMARY_QUERY=`cat<<EOF
SELECT 
  matchid,
  second,
  periodid,
  playerid,
  teamid,
  position,
  section_x,
  section_y,
  SUM(if (section_x < prev_section_x, 1, 0))  AS direction_x_2,
  SUM(if (section_x == prev_section_x, 1, 0)) AS direction_x_1,
  SUM(if (section_x > prev_section_x, 1, 0))  AS direction_x_0,
  SUM(if (section_y < prev_section_y, 1, 0))  AS direction_y_2,
  SUM(if (section_y == prev_section_y, 1, 0)) AS direction_y_1,
  SUM(if (section_y > prev_section_y, 1, 0))  AS direction_y_0,
  COUNT(section_x) AS total_x,
  COUNT(section_y) AS total_y
FROM 
  (SELECT
    lhs.second AS second,
    lhs.matchid AS matchid,
    lhs.periodid AS periodid,
    lhs.playerid AS playerid,
    lhs.teamid AS teamid,
    position,
    if (x <= 16.6, 0, if(x <= 33.2, 1, if(x <= 49.9, 2, if(x <= 66.4, 3, if(x <= 83, 4, 5))))) AS section_x,
    if (y <= 33, 0, if(y <= 66, 1, 2)) AS section_y, 
    if (prev_x == NULL, x, if(prev_x <= 16.6, 0, if(prev_x <= 33.2, 1, if(prev_x <= 49.9, 2, if(prev_x <= 66.4, 3, if(prev_x <= 83, 4, 5)))))) AS prev_section_x,
    if (prev_y == NULL, y, if(prev_y <= 33, 0, if(prev_x <= 66, 1, 2))) AS prev_section_y
  FROM 
    (SELECT
      INTEGER(FLOOR(TIMESTAMP_TO_MSEC(timestamp)/1000)) AS second,
      periodid,
      playerid,
      matchid, 
      teamid,
      x, 
      y,
      LEAD(x, 1) OVER (PARTITION BY playerid, matchid ORDER BY second DESC) as prev_x,
      LEAD(y, 1) OVER (PARTITION BY playerid, matchid ORDER BY second DESC) as prev_y
    FROM [toque.touches] WHERE playerid != 0 GROUP EACH BY x, y, playerid, matchid, teamid, second, periodid) AS lhs
      JOIN (SELECT matchid, teamid, playerid, position FROM [$POSITION_SUMMARY]) AS rhs 
        ON lhs.matchid=rhs.matchid AND lhs.teamid=rhs.teamid AND lhs.playerid=rhs.playerid)
GROUP EACH BY
  matchid,
  second,
  periodid,
  playerid,
  teamid,
  position,
  section_x,
  section_y
ORDER EACH BY
  matchid,
  second,
  periodid
EOF`

MOVE_SUMMARY=$DATASET.move_summary
MOVE_SUMMARY_JOB=MOVE_SUMMARY_$SECONDS_$RANDOM_$RANDOM_$$
echo "Generating $MOVE_SUMMARY"
bq \
  --project_id=$PROJECTID \
  --synchronous_mode=false \
  --job_id=$MOVE_SUMMARY_JOB \
    query \
      -n 0 \
      --allow_large_results \
      --replace \
      --destination_table $MOVE_SUMMARY \
        "$MOVE_SUMMARY_QUERY"


bq \
  --project_id=$PROJECTID \
    wait $MOVE_SUMMARY_JOB
bq \
  --project_id=$PROJECTID \
    wait $GAME_SUMMARY_JOB
