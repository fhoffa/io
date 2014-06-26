from pandas.io import gbq

touch_table = 'cloude-sandbox:toque.touches'
match_game_table = 'SELECT * FROM [cloude-sandbox:temp.match_games_with_replacement]'
match_goals_table = 'SELECT * FROM [cloude-sandbox:temp.match_goals_table_20140623]'

# Number of games to look at history from:
history_size = 3

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

expected_goals_match = """
SELECT matchid, teamid,
  COUNT(eventid) as sot,
  SUM(typeid = 16) as goals,
  AVG(dist) as avgdist,
  sum(pG) as xG
FROM (SELECT *, 
  IF(dist = 1, 1, 1.132464 - 0.303866* LN(dist)) as pG
FROM (SELECT matchid, teamid, playerid, eventid, typeid, x, y,
      ROUND(SQRT(POW((100-x),2) + POW((50-y),2))) as dist,
      IF(typeid = 16,1,0) as goal
FROM (SELECT matchid, teamid, playerid, eventid, typeid, x, y,
    SUM(qualifiers.type = 82) WITHIN RECORD AS blck,
    SUM(qualifiers.type IN (26,9)) WITHIN RECORD AS penfk,
    SUM(qualifiers.type = 28) WITHIN RECORD AS og,
   FROM [%(touch_table)s] 
   WHERE typeid IN (14, 15,16))
WHERE blck = 0 AND og = 0 AND penfk = 0)
WHERE dist < 40)
GROUP BY matchid, teamid
ORDER BY matchid, teamid
   """ % {'touch_table': touch_table}


own_goals_by_team_subquery = """
SELECT matchid, teamid, count(*) as own_goals
FROM [%(touch_table)s] 
WHERE typeid = %(goal)d AND qualifiers.type = %(own_goal)d 
GROUP BY matchid, teamid
""" % {'touch_table': touch_table,
       'goal': goal_id, 
       'own_goal': own_goal_qualifier_id}

pass_stats = """
SELECT matchid, teamid, SUM(pass_80) as pass_80, SUM(pass_70) as pass_70
FROM (
   SELECT matchid, teamid, outcomeid, 
   if (x > 80 and outcomeid = 1, 1, 0) as pass_80,
   if (x > 70 and outcomeid = 1, 1, 0) as pass_70 
  FROM [%(touch_table)s] WHERE typeid = %(pass)s)
GROUP BY matchid, teamid
""" % {'touch_table' : touch_table,
       'pass': pass_id}

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
match_goals_subquery = """
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
FROM (%(match_games)s) 
)h
JOIN (%(match_games)s) m1
ON h.teamid = m1.teamid
WHERE
h.tenth_last_match_timestamp is not NULL AND
h.last_match_timestamp IS NOT NULL AND
m1.timestamp >= h.tenth_last_match_timestamp AND 
m1.timestamp <= h.last_match_timestamp 

""" % {'history_size': history_size, 
       'match_games': match_game_table}
        

team_game_summary = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes as passes,
t.bad_passes as bad_passes,
t.passes / (t.passes + t.bad_passes + 1) as pass_ratio,
t.corners as corners,
t.fouls as fouls,
t.shots as shots,
t.cards as cards,
p.pass_80 as pass_80,
p.pass_70 as pass_70,
TIMESTAMP_TO_MSEC(t.timestamp) as timestamp,
g.goals as goals,
h.is_home as is_home,
h.team_name as team_name,
h.competitionid as competitionid,
if (x.xG is not null, x.xG, 0.0) as expected_goals,
if (x.sot is not null, x.xG, 0.0) as on_target,
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
    FROM [%(touch_table)s]  as t    
    WHERE teamid != 0)    
 GROUP BY matchid, teamid 
) t
LEFT OUTER JOIN (%(match_goals)s) as g
ON t.matchid = g.matchid and t.teamid = g.teamid
JOIN
(%(pass_stats)s) p
ON
t.matchid = p.matchid and t.teamid = p.teamid
JOIN  (%(match_games)s) h
ON t.matchid = h.matchid AND t.teamid = h.teamid
LEFT OUTER JOIN (%(expected_goals)s) x
ON t.matchid = x.matchid AND t.teamid = x.teamid
-- ORDER BY matchid, is_home
""" % {'pass': pass_id,
       'foul': foul_id,
       'corner': corner_id,
       'half': half_id,
       'shots': ','.join([str(id) for id in shot_ids]),
       'card': card_id,
       'pass_stats': pass_stats,
       'expected_goals': expected_goals_match,
       'touch_table': touch_table,
       'match_games': match_game_table,
       'match_goals': match_goals_table}
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
  cur.team_name as team_name,
  cur.pass_80 as pass_80,
  cur.pass_70 as pass_70,
  cur.expected_goals as expected_goals,
  cur.on_target as on_target,
  
  opp.teamid as op_teamid,
  opp.passes as op_passes,
  opp.bad_passes as op_bad_passes,
  opp.pass_ratio as op_pass_ratio,
  opp.corners as op_corners,
  opp.fouls as op_fouls,
  opp.cards as op_cards, 
  opp.goals as op_goals,
  opp.shots as op_shots,
  opp.team_name as op_team_name,
  opp.pass_80 as op_pass_80,
  opp.pass_70 as op_pass_70,
  opp.expected_goals as op_expected_goals,
  opp.on_target as op_on_target,

  cur.competitionid as competitionid,

  if (opp.shots > 0, cur.shots / opp.shots, cur.shots * 1.0) as shots_op_ratio,
  if (opp.goals > 0, cur.goals / opp.goals, cur.goals * 1.0) as goals_op_ratio,
  if (opp.pass_ratio > 0, cur.pass_ratio / opp.pass_ratio, 1.0) as pass_op_ratio,
 
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
pts.competitionid as competitionid,
pts.is_home as is_home,
pts.team_name as team_name,
pts.op_team_name as op_team_name,
pts.timestamp as timestamp,

summary.total_points as total_points,
summary.total_goals as total_goals,
summary.total_op_goals as total_op_goals,

summary.pass_70 as pass_70,
summary.pass_80 as pass_80,
summary.op_pass_70 as op_pass_70,
summary.op_pass_80 as op_pass_80,
summary.expected_goals as expected_goals,
summary.op_expected_goals as op_expected_goals,
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

summary.goals_op_ratio as goals_op_ratio,
summary.shots_op_ratio as shots_op_ratio,
summary.pass_op_ratio as pass_op_ratio,

FROM (
SELECT hist.matchid as matchid,
  hist.teamid as teamid,

  AVG(games.pass_70) as pass_70, 
  AVG(games.pass_80) as pass_80, 
  AVG(games.op_pass_70) as op_pass_70, 
  AVG(games.op_pass_80) as op_pass_80, 
  AVG(games.expected_goals) as expected_goals, 
  AVG(games.op_expected_goals) as op_expected_goals, 
  AVG(games.passes) as passes, 
  AVG(games.bad_passes) as bad_passes, 
  AVG(games.pass_ratio) as pass_ratio,
  AVG(games.corners) as corners, 
  AVG(games.fouls) as fouls,
  AVG(games.cards) as cards, 
  SUM(games.goals) as total_goals, 
  SUM(games.points) as total_points, 
  AVG(games.shots) as shots,
  AVG(games.op_passes) as op_passes, 
  AVG(games.op_bad_passes) as op_bad_passes, 
  AVG(games.op_corners) as op_corners,
  AVG(games.op_fouls) as op_fouls,
  AVG(games.op_cards) as op_cards,   
  AVG(games.op_shots) as op_shots, 
  SUM(games.op_goals) as total_op_goals, 
  AVG(games.goals_op_ratio) as goals_op_ratio,
  AVG(games.shots_op_ratio) as shots_op_ratio,
  AVG(games.pass_op_ratio) as pass_op_ratio,
  
FROM (%(match_history)s)  hist
JOIN (%(team_game_op_summary)s) games
ON hist.previous_match = games.matchid and
 hist.teamid = games.teamid
GROUP BY matchid, teamid
) as summary
JOIN (%(match_games)s) pts on summary.matchid = pts.matchid
and summary.teamid = pts.teamid
WHERE summary.matchid <> '442291'
ORDER BY matchid, is_home DESC
""" % {'team_game_op_summary': team_game_op_summary,
       'match_games': match_game_table,
       'match_history': match_history}

history_query_with_goals= """
SELECT   

h.matchid as matchid,
h.teamid as teamid,
h.op_teamid as op_teamid,
h.competitionid as competitionid,
h.is_home as is_home,
h.team_name as team_name,
h.op_team_name as op_team_name,
h.timestamp as timestamp,

g.goals as goals,
op.goals as op_goals,
if (g.goals > op.goals, 3,
  if (g.goals == op.goals, 1, 0)) as points,

h.total_points as total_points,
h.total_goals as total_goals,
h.total_op_goals as total_op_goals,

h.pass_70 as pass_70,
h.pass_80 as pass_80,
h.op_pass_70 as op_pass_70,
h.op_pass_80 as op_pass_80,
h.expected_goals as expected_goals,
h.op_expected_goals as op_expected_goals,
h.passes as passes,
h.bad_passes as bad_passes,
h.pass_ratio as pass_ratio,
h.corners as corners,
h.fouls as fouls,
h.cards as cards,
h.shots as shots,

h.op_passes as op_passes,
h.op_bad_passes as op_bad_passes,
h.op_corners as op_corners,
h.op_fouls as op_fouls,
h.op_cards as op_cards,
h.op_shots as op_shots,

h.goals_op_ratio as goals_op_ratio,
h.shots_op_ratio as shots_op_ratio,
h.pass_op_ratio as pass_op_ratio,

FROM (%(history_query)s) h
JOIN (%(match_goals)s) g
ON h.matchid = g.matchid and h.teamid = g.teamid
JOIN (%(match_goals)s) op
ON h.matchid = op.matchid and h.op_teamid = op.teamid
""" % {'history_query': history_query,
       'match_goals': match_goals_table
      }

wc_history_query = """
SELECT * FROM (%(history_query)s) WHERE competitionid = 4
AND timestamp > '2014-01-01 00:00:00.000000'
""" % {'history_query': history_query,
      }

def get_wc_features():
  return gbq.read_gbq(wc_history_query)

def get_features():
  return gbq.read_gbq(history_query_with_goals)

def get_non_feature_columns():
  return ['teamid', 'op_teamid', 'matchid', 'competitionid',
          'goals', 'op_goals', 'points', 'timestamp', 'team_name', 
          'op_team_name']

def get_feature_columns(all_cols):
  return [col for col in all_cols if col not in get_non_feature_columns()]

