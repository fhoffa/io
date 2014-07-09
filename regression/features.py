from pandas.io import gbq

import match_stats

# Number of games to look at history from in order to predict future performance. 
# After the group phase (when we did the initial predictions) we set this value s to 3,
# since we had 3 games of history in the group phase for each team.
history_size = 5

# Games that have stats available. Not all games in the match_games table
# will have stats (e.g. they might be in the future).
match_game_with_stats = """
    SELECT * FROM (%(match_games)s)
    WHERE matchid in (
      SELECT matchid FROM (%(stats_table)s) GROUP BY matchid)
    """ % {'match_games': match_stats.match_games_table(),
           'stats_table': match_stats.team_game_summary_query()}        

# For each team t in each game g, computes the N previous game ids where team t
# played, where N is the history_size (number of games of history we
# use for prediction). The statistics of the N previous games will be used
# to predict the outcome of game g.
match_history = """
SELECT h.teamid as teamid, h.matchid as matchid, h.timestamp as timestamp, 
m1.timestamp as previous_timestamp, m1.matchid as previous_match
FROM (
SELECT teamid, matchid, timestamp, 
LEAD(matchid, 1) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as last_matchid,
LEAD(timestamp, 1) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as last_match_timestamp,
LEAD(timestamp, %(history_size)d) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as nth_last_matchid,
LEAD(timestamp, %(history_size)d) OVER (PARTITION BY teamid ORDER BY timestamp DESC) as nth_last_match_timestamp,
FROM (%(match_games)s) 
) h
JOIN (%(match_games_with_stats)s) m1
ON h.teamid = m1.teamid
WHERE
h.nth_last_match_timestamp is not NULL AND
h.last_match_timestamp IS NOT NULL AND
m1.timestamp >= h.nth_last_match_timestamp AND 
m1.timestamp <= h.last_match_timestamp 

""" % {'history_size': history_size, 
       'match_games_with_stats': match_game_with_stats,
       'match_games': match_stats.match_games_table()}

# Computes summary statistics for the N preceeding matches.
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

summary.avg_points as avg_points,
summary.avg_goals as avg_goals,
summary.op_avg_goals as op_avg_goals,

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
  AVG(games.goals) as avg_goals, 
  AVG(games.points) as avg_points, 
  AVG(games.shots) as shots,
  AVG(games.op_passes) as op_passes, 
  AVG(games.op_bad_passes) as op_bad_passes, 
  AVG(games.op_corners) as op_corners,
  AVG(games.op_fouls) as op_fouls,
  AVG(games.op_cards) as op_cards,   
  AVG(games.op_shots) as op_shots, 
  AVG(games.op_goals) as op_avg_goals, 
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
""" % {'team_game_op_summary': match_stats.team_game_op_summary_query(),
       'match_games': match_stats.match_games_table(),
       'match_history': match_history}

# Expands the history_query, which summarizes statistics from past games
# with the result of who won the current game. This information will not
# be availble for future games that we want to predict, but it will be
# available for past games. We can then use this information to train our
# models.
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

h.avg_points as avg_points,
h.avg_goals as avg_goals,
h.op_avg_goals as op_avg_goals,

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
       'match_goals': match_stats.match_goals_table()
      }

# Identical to the history_query (which, remember, does not have
# outcomes).
wc_history_query = """
SELECT * FROM (%(history_query)s) WHERE competitionid = 4
""" % {'history_query': history_query,
      }

# Runs a bigquery query that gets the features that can be used
# to predict the world cup.
def get_wc_features():
  return gbq.read_gbq(wc_history_query)

# Runs a BigQuery query to get features that can be used to train
# a machine learning model.
def get_features():
  return gbq.read_gbq(history_query_with_goals)

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

