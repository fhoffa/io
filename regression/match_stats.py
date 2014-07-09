# Raw touch-by-touch BigQuery table. This table contains data licensed from Opta
# and so cannot be shared widely.
_touch_table = 'cloude-sandbox:toque.touches'
# Set the touch table to to use the summary table.
# _touch_table = None

# Table containing games that were played. Allows us to map team ids to team
# names and figure out which team was home and which team was away.
_match_games_table = """
    SELECT * FROM [cloude-sandbox:temp.match_games_with_replacement]
"""

# Computing the score of the game is actually fairly tricky to do from the
# touches table, so we use this view to keep track of game score.
_match_goals_table = """
    SELECT * FROM [cloude-sandbox:public.match_goals_table_20140707]
"""

# View that computes the statistics (features) about the games.
game_summary = """
    SELECT * FROM [cloude-sandbox:public.team_game_summary_20140707]
"""

# Number of games to look at history from in order to predict future performance. 
# After the group phase (when we did the initial predictions) we set this value s to 3,
# since we had 3 games of history in the group phase for each team.
history_size = 5

# Event type ids from the touches table.
_pass_id = 1
_foul_id = 4
_corner_id = 6
_shot_ids = [13, 14, 15, 16]
_goal_id = 16
_card_id = 17
_half_id = 32
_game_id = 34

# Qualifiers
_own_goal_qualifier_id = 28

# Computes the expected goal statistic, based on shot location.
_expected_goals_match = """
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
   """ % {'touch_table': _touch_table}

# Subquery to compute raw number of goals scored. Does not take
# into account own-goals (i.e. if a player scores an own-goal against
# his own team, it will count as a goal for that team.
# Computes passing statistics. 
# pass_80: Number of passes completed in the attacking fifth of the field.
# pass_70: NUmber of passes completed in the attacking third of the field.
_pass_stats = """
SELECT matchid, teamid, SUM(pass_80) as pass_80, SUM(pass_70) as pass_70
FROM (
   SELECT matchid, teamid, outcomeid, 
   if (x > 80 and outcomeid = 1, 1, 0) as pass_80,
   if (x > 70 and outcomeid = 1, 1, 0) as pass_70 
  FROM [%(touch_table)s] WHERE typeid = %(pass)s)
GROUP BY matchid, teamid
""" % {'touch_table' : _touch_table,
       'pass': _pass_id}

# Subquery that tracks own goals so we can later attribute them to
# the other team.
_own_goals_by_team_subquery = """
SELECT matchid, teamid, count(*) as own_goals
FROM [%(touch_table)s] 
WHERE typeid = %(goal)d AND qualifiers.type = %(own_goal)d 
GROUP BY matchid, teamid
""" % {'touch_table': _touch_table,
       'goal': _goal_id, 
       'own_goal': _own_goal_qualifier_id}

# Subquery that credits an own goal to the opposite team.
_own_goal_credit_subquery = """
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
""" % {'touch_table': _touch_table,
       'own_goals': _own_goals_by_team_subquery}

# Simple query that computes the number of goals in a game
# (not counting penalty kicks that occur after a draw).
# This is not sufficient to determine the score, since the
# data will attribute own goals to the wrong team.
_raw_goal_and_game_subquery = """
SELECT  matchid, teamid, goal, game, timestamp,
FROM (
  SELECT matchid, teamid, 
    if (typeid == %(goal)d and periodid != 5, 1, 0) as goal,      
    if (typeid == %(game)d, 1, 0) as game,
    eventid,
    timestamp,
  FROM [%(touch_table)s]
  WHERE typeid in (%(goal)d, %(game)d))
""" % {'goal': _goal_id, 
       'game': _game_id,
       'touch_table': _touch_table}

# Score by game and team, not adjusted for own goals.
_raw_goal_by_game_and_team_subquery = """
SELECT matchid, teamid, SUM(goal) as goals, 
    MAX(TIMESTAMP_TO_USEC(timestamp)) as timestamp,
FROM (%s)
GROUP BY matchid, teamid
""" % (_raw_goal_and_game_subquery)

# Compute the number of goals in the game. To do this, we want to subtract off
# any own goals a team scored against themselves, and add the own goals that a
# team's opponent scored.
_match_goals_subquery = """
SELECT matchid, teamid , goals + delta as goals, timestamp as timestamp 
FROM (
    SELECT goals.matchid as matchid , goals.teamid as teamid,
        goals.goals as goals,
        goals.timestamp as timestamp,
        if (cr.cnt is not NULL, INTEGER(cr.cnt), INTEGER(0)) 
            - if (de.cnt is not NULL, INTEGER(de.cnt), INTEGER(0)) as delta
    FROM (%s) goals
    LEFT OUTER JOIN (%s) cr
    ON goals.matchid = cr.matchid and goals.teamid = cr.credit_team
    LEFT OUTER JOIN (%s) de
    ON goals.matchid = de.matchid and goals.teamid = de.deduct_team
)
""" % (_raw_goal_by_game_and_team_subquery,
       _own_goal_credit_subquery,
       _own_goal_credit_subquery)

# Query that summarizes statistics by team and by game.
# Statistics computed are:
# passes: number of passes per minute completed in the game.
# bad_passes: number of passes per minute that were not completed.
# pass_ratio: proportion of passes that were completed.
# corners: number of corner kicks awarded per minute.
# shots: number of shots taken per minute.
# fouls: number of fouls committed per minute.
# cards: number of cards (yellow or red) against members of the team.
# pass_{70,80}: number of completed passes per minute in the attacking {70,80%}
#     zone.
# is_home: whether this game was played at home.
# expected_goals: number of goals expected given the number and location 
#     of shots on goal.
# on_target: number of shots on target per minute
_team_game_summary = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes / t.length as passes,
t.bad_passes / t.length as bad_passes,
t.passes / (t.passes + t.bad_passes + 1) as pass_ratio,
t.corners / t.length as corners,
t.fouls / t.length  as fouls,
t.shots / t.length  as shots,
t.cards as cards,
p.pass_80 / t.length as pass_80,
p.pass_70 / t.length as pass_70,
TIMESTAMP_TO_MSEC(t.timestamp) as timestamp,
g.goals as goals,
h.is_home as is_home,
h.team_name as team_name,
h.competitionid as competitionid,
if (x.xG is not null, x.xG, 0.0) as expected_goals,
if (x.sot is not null, INTEGER(x.sot), INTEGER(0)) / t.length as on_target,
t.length as length
FROM (
 SELECT matchid, teamid,
      sum(pass) as passes,
      sum(bad_pass) as bad_passes,
      sum (corner) as corners,
      sum (foul) as fouls,      
      sum(shots) as shots,
      sum(cards) as cards,
      max(timestamp) as timestamp,
      max([min]) as length,
      1  as games,     
  FROM (
    SELECT matchid, teamid,       
      timestamp, [min],
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
""" % {'pass': _pass_id,
       'foul': _foul_id,
       'corner': _corner_id,
       'half': _half_id,
       'shots': ','.join([str(id) for id in _shot_ids]),
       'card': _card_id,
       'pass_stats': _pass_stats,
       'expected_goals': _expected_goals_match,
       'touch_table': _touch_table,
       'match_games': _match_games_table,
       'match_goals': _match_goals_table}


# Some of the games in the touches table have been ingested twice. If that
# is the case, scale the game statistics.
_team_game_summary_corrected = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes / s.event_count as passes,
t.bad_passes / s.event_count as bad_passes,
t.pass_ratio as pass_ratio,
t.corners / s.event_count as corners,
t.fouls / s.event_count as fouls,
t.shots / s.event_count as shots,
t.cards / s.event_count as cards,
t.pass_80 / s.event_count as pass_80,
t.pass_70 / s.event_count as pass_70,
t.timestamp as timestamp,
t.goals / s.event_count as goals,
t.is_home as is_home,
t.team_name as team_name,
t.competitionid as competitionid,
t.expected_goals / s.event_count as expected_goals,
t.on_target / s.event_count as on_target,
t.length as length,

FROM  (%(team_game_summary)s) t
JOIN (
SELECT matchid, MAX(event_count) as event_count
FROM (
    SELECT matchid, COUNT(eventid) as event_count  
    FROM [%(touches_table)s]
    GROUP EACH BY matchid, eventid
) GROUP BY matchid
) s ON t.matchid = s.matchid
""" % {
       'team_game_summary': _team_game_summary,
       'touches_table': _touch_table}

# Combines statistics from both teams in a match.
# For each two records matching the pattern (m, t1, <stats1>) and 
# (m, t2, <stats2>) where m is the match id, t1 and t2 are the two teams,
# stats1 and stats2 are the statistics for those two teams, combines them
# into a single row (m, t1, t2, <stats1>, <stats2>) where all of the 
# t2 field names are decorated with the op_ prefix. For example, teamid becomes
# op_teamid, and pass_70 becomes op_pass_70.
_team_game_op_summary =  """
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
      cur.length as length,
      
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

      if (opp.shots > 0, cur.shots / opp.shots, cur.shots * 1.0)
	  as shots_op_ratio,
      if (opp.goals > 0, cur.goals / opp.goals, cur.goals * 1.0)
	  as goals_op_ratio,
      if (opp.pass_ratio > 0, cur.pass_ratio / opp.pass_ratio, 1.0)
	  as pass_op_ratio,
     
      if (cur.goals > opp.goals, 3,
	if (cur.goals == opp.goals, 1, 0)) as points,
      cur.timestamp as timestamp,

    FROM (%(team_game_summary)s) cur
    JOIN (%(team_game_summary)s) opp 
    ON cur.matchid = opp.matchid
    WHERE cur.teamid != opp.teamid
    ORDER BY cur.matchid, cur.teamid
      """ % {'team_game_summary': _team_game_summary_corrected}

###
### Public queries / methods
###

# Query that returns query statistics for both teams in a game.
def team_game_op_summary_query(): return _team_game_op_summary
def team_game_summary_query(): return _team_game_summary_corrected
def match_goals_table(): return _match_goals_table
def match_games_table(): return _match_games_table

# Returns a list of the columns that are in our features dataframe that
# should not be used in prediction. These are essentially either metadata
# columns (team name, for example), or potential target variables that
# include the outcome. We want to make sure not to use the latter, since
# we don't want to use information about the current game to predict that
# same game.
def get_non_feature_columns():
  return ['teamid', 'op_teamid', 'matchid', 'competitionid',
          'goals', 'op_goals', 'points', 'timestamp', 'team_name', 
          'op_team_name']

# Returns a list of all columns that should be used in prediction (i.e. all
# features that are in the dataframe but are not in the 
# features.get_non_feature_column() list).
def get_feature_columns(all_cols):
  return [col for col in all_cols if col not in get_non_feature_columns()]

