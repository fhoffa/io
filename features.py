from pandas.io import gbq

touch_table = 'cloude-sandbox:toque.touches'

# Number of games to look at history from:
history_size = 10

# Event type ids:
pass_id = 1
foul_id = 4
corner_id = 6
shot_ids = [13, 14, 15, 16]
goal_id = 16
card_id = 17
half_id = 32
game_id = 34

# Qualifiers
own_goal_qualifier_id = 28

own_goals_by_team_subquery = """
SELECT matchid, teamid, count(*) as own_goals
FROM [%(touch_table)s] 
WHERE typeid = %(goal)d AND qualifiers.type = %(own_goal)d 
GROUP BY matchid, teamid
""" % {'touch_table': touch_table,
       'goal': goal_id, 
       'own_goal': own_goal_qualifier_id}

own_goal_credit_subquery = """
SELECT games.matchid as matchid, 
  games.teamid as credit_team,
  og.teamid as deduct_team,
  og.own_goals as cnt
FROM 
(%(own_goals)s) og
JOIN (
  SELECT matchid, teamid, 
  FROM [%(touch_table)s]
  GROUP BY matchid, teamid) games
ON og.matchid = games.matchid 
WHERE games.teamid <> og.teamid
""" % {'touch_table': touch_table,
       'own_goals': own_goals_by_team_subquery}

# Subquery to compute raw number of goals scored. Does not take
# into account own-goals (i.e. if a player scores an own-goal against
# his own team, it will count as a goal for that team.
raw_goal_and_game_subquery = """
SELECT  matchid, teamid, goal, game, timestamp,
FROM (
  SELECT matchid, teamid, 
    if (typeid == %(goal)d, 1, 0) as goal,      
    if (typeid == %(game)d, 1, 0) as game,
    eventid,
    timestamp,
  FROM [%(touch_table)s]
  WHERE typeid in (%(goal)d, %(game)d))
""" % {'goal': goal_id, 
       'game': game_id,
       'touch_table': touch_table}

raw_goal_by_game_and_team_subquery = """
SELECT matchid, teamid, sum(goal) as goals, max(TIMESTAMP_TO_USEC(timestamp)) as timestamp,
FROM (%s)
GROUP BY matchid, teamid
""" % (raw_goal_and_game_subquery)

# Compute the number of goals in the game. To do this, we want to subtract off any own goals
# a team scored against themselves, and add the own goals that a team's opponent scored.
match_goals = """
SELECT matchid, teamid , goals + delta as goals, timestamp as timestamp 
FROM (
SELECT goals.matchid as matchid , goals.teamid as teamid, goals.goals as goals, goals.timestamp as timestamp,
if (cr.cnt is not NULL, INTEGER(cr.cnt), INTEGER(0)) 
- if (de.cnt is not NULL, INTEGER(de.cnt), INTEGER(0)) as delta

FROM (%s) goals
LEFT OUTER JOIN (%s) cr on
goals.matchid = cr.matchid and goals.teamid = cr.credit_team
LEFT OUTER JOIN (%s) de on 
goals.matchid = de.matchid and goals.teamid = de.deduct_team
)
--ORDER BY matchid, teamid
""" % (raw_goal_by_game_and_team_subquery, own_goal_credit_subquery, own_goal_credit_subquery)

match_history = """
SELECT h.teamid as teamid, h.matchid as matchid, h.timestamp as timestamp, 
m1.timestamp as previous_timestamp, m1.matchid as previous_match
FROM (
SELECT teamid, matchid, timestamp, 
LEAD(matchid, 1) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as last_matchid,
LEAD(timestamp, 1) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as last_match_timestamp,
LEAD(timestamp, %(history_size)d) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as tenth_last_matchid,
LEAD(timestamp, %(history_size)d) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as tenth_last_match_timestamp,
FROM (%(match_goals)s) 
)h
JOIN (%(match_goals)s) m1
ON h.teamid = m1.teamid
WHERE
h.tenth_last_match_timestamp is not NULL AND
h.last_match_timestamp IS NOT NULL AND
m1.timestamp >= h.tenth_last_match_timestamp AND 
m1.timestamp <= h.last_match_timestamp 

""" % {'history_size': history_size, 
       'match_goals': match_goals}

team_game_summary = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes as passes,
t.bad_passes as bad_passes,
t.passes / (t.passes + t.bad_passes) as pass_ratio,
t.corners as corners,
t.fouls as fouls,
t.shots as shots,
t.cards as cards,
TIMESTAMP_TO_MSEC(t.timestamp) as timestamp,
g.goals as goals,
h.is_home as is_home,
FROM (
 SELECT matchid, teamid,
      sum(pass) as passes,
      sum(bad_pass) as bad_passes,
      sum (corner) as corners,
      sum (foul) as fouls,      
      sum(shots) as shots,
      sum(cards) as cards,
      max(timestamp) as timestamp,
      1  as games,     
  FROM (
    SELECT matchid, teamid,       
      timestamp,
      if (typeid == %(pass)d and outcomeid = 1, 1, 0) as pass,
      if (typeid == %(pass)d and outcomeid = 0, 1, 0) as bad_pass,
      if (typeid == %(foul)d and outcomeid = 1, 1, 0) as foul,
      if (typeid == %(corner)d and outcomeid = 1, 1, 0) as corner,
      if (typeid == %(half)d, 1, 0) as halves,
      if (typeid in (%(shots)s), 1, 0) as shots,
      if (typeid == %(card)d, 1, 0) as cards,                             
    FROM [toque.touches]  as t    
    WHERE teamid != 0)    
 GROUP BY matchid, teamid 
) t
LEFT OUTER JOIN (%(match_goals)s) as g
ON t.matchid = g.matchid and t.teamid = g.teamid
JOIN
(SELECT * FROM
(SELECT INTEGER(SUBSTR(hometeam_id, 2)) teamid, hometeam_name team_name, STRING(gameid) matchid, 1 is_home
FROM [toque.matches]),
(SELECT INTEGER(SUBSTR(awayteam_id, 2)) teamid, awayteam_name team_name, STRING(gameid) matchid, 0 is_home
FROM [toque.matches])
) h
ON t.matchid = h.matchid AND t.teamid = h.teamid
ORDER BY matchid, is_home
""" % {'pass': pass_id,
       'foul': foul_id,
       'corner': corner_id,
       'half': half_id,
       'shots': ','.join([str(id) for id in shot_ids]),
       'card': card_id,
       'match_goals': match_goals}
# print team_game_summary

team_game_op_summary = """
SELECT cur.matchid as matchid,
  cur.teamid as teamid,
  cur.passes as passes,
  cur.bad_passes as bad_passes,
  cur.pass_ratio as pass_ratio,
  cur.corners as corners,
  cur.fouls as fouls,  
  cur.cards as cards,
  cur.goals as goals,
  cur.shots as shots,
  cur.is_home as is_home,
  
  opp.teamid as op_teamid,
  opp.passes as op_passes,
  opp.bad_passes as op_bad_passes,
  opp.pass_ratio as op_pass_ratio,
  opp.corners as op_corners,
  opp.fouls as op_fouls,
  opp.cards as op_cards, 
  opp.goals as op_goals,
  opp.shots as op_shots,
 
  if (cur.goals > opp.goals, 3,
    if (cur.goals == opp.goals, 1, 0)) as points,
  cur.timestamp as timestamp,

FROM (%(team_game_summary)s) cur
JOIN (%(team_game_summary)s) opp 
ON cur.matchid = opp.matchid
WHERE cur.teamid != opp.teamid
ORDER BY cur.matchid, cur.teamid
""" % {'team_game_summary': team_game_summary}

history_query = """

SELECT  
summary.matchid as matchid,
pts.teamid as teamid,
pts.op_teamid as op_teamid,
pts.points as points,
pts.goals as goals,
pts.op_goals as op_goals,
pts.is_home as is_home,

/*
summary.total_points as total_points,
summary.total_goals as total_goals,
summary.total_op_goals as total_op_goals,
*/

summary.home_passes as home_passes,
summary.home_goals as home_goals,
summary.home_shots as home_shots,

/*
summary.away_passes as away_passes,
summary.away_goals as away_goals,
summary.away_shots as away_shots,
*/
summary.op_home_passes as op_home_passes,
summary.op_home_goals as op_home_goals,
summary.op_home_shots as op_home_shots,

summary.op_away_passes as op_away_passes,
summary.op_away_goals as op_away_goals,
summary.op_away_shots as op_away_shots,

summary.passes as passes,
summary.bad_passes as bad_passes,
summary.pass_ratio as pass_ratio,
summary.corners as corners,
summary.fouls as fouls,
summary.cards as cards,
summary.shots as shots,

summary.op_passes as op_passes,
summary.op_bad_passes as op_bad_passes,
summary.op_corners as op_corners,
summary.op_fouls as op_fouls,
summary.op_cards as op_cards,
summary.op_shots as op_shots,

FROM (
SELECT hist.matchid as matchid,
  hist.teamid as teamid,
  SUM(IF(games.is_home > 0, games.passes, 0)) as home_passes,
  SUM(IF(games.is_home > 0, games.goals, 0)) as home_goals,
  SUM(IF(games.is_home > 0, games.shots, 0)) as home_shots,
  
  SUM(IF(games.is_home == 0, games.passes, 0)) as away_passes,
  SUM(IF(games.is_home == 0, games.goals, 0)) as away_goals,
  SUM(IF(games.is_home == 0, games.shots, 0)) as away_shots,

  SUM(IF(games.is_home > 0, games.op_passes, 0)) as op_home_passes,
  SUM(IF(games.is_home > 0, games.op_goals, 0)) as op_home_goals,
  SUM(IF(games.is_home > 0, games.op_shots, 0)) as op_home_shots,

  SUM(IF(games.is_home == 0, games.op_passes, 0)) as op_away_passes,
  SUM(IF(games.is_home == 0, games.op_goals, 0)) as op_away_goals,
  SUM(IF(games.is_home == 0, games.op_shots, 0)) as op_away_shots,

  AVG(games.passes) as passes, 
  AVG(games.bad_passes) as bad_passes, 
  AVG(games.pass_ratio) as pass_ratio,
  AVG(games.corners) as corners, 
  AVG(games.fouls) as fouls,
  AVG(games.cards) as cards, 
  SUM(games.goals) as total_goals, 
  AVG(games.shots) as shots,
  AVG(games.op_passes) as op_passes, 
  AVG(games.op_bad_passes) as op_bad_passes, 
  AVG(games.op_corners) as op_corners,
  AVG(games.op_fouls) as op_fouls,
  AVG(games.op_cards) as op_cards,   
  AVG(games.op_shots) as op_shots, 
  SUM(games.op_goals) as total_op_goals, 
  SUM(games.points) as total_points,  
  
FROM (%(match_history)s)  hist
JOIN (%(team_game_op_summary)s) games
ON hist.previous_match = games.matchid and
 hist.teamid = games.teamid
GROUP BY matchid, teamid
) as summary
JOIN (%(team_game_op_summary)s) pts on summary.matchid = pts.matchid
and summary.teamid = pts.teamid
WHERE summary.matchid <> '442291'
ORDER BY matchid, is_home
""" % {'team_game_op_summary': team_game_op_summary,
       'match_history': match_history}

def get_features():
  return gbq.read_gbq(history_query)

