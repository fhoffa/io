"""
    Ranks soccer teams by computing a power index based
    on game outcomes.
"""

import world_cup

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from pandas.io import gbq
import pylab as pl
import random
import scipy.cluster
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import statsmodels.api as sm

def buildTeamMatrix(data, target_col):
  teams = {}
  n = len(data) / 2
  for teamid in data['teamid']:
    teams[str(teamid)] = pd.Series(np.zeros(n))

  result = pd.Series(np.empty(n))
  teams[target_col] = result

  current_season = None
  current_discount = 1.0

  for game in xrange(n):
    home = data.iloc[game * 2]
    away = data.iloc[game * 2 + 1]
    if home['seasonid'] <> current_season:
      # Discount older seasons.
      current_season = home['seasonid']
      current_discount *= 0.9
      print "New season %s" % (current_season,)

    home_id = str(home['teamid'])
    away_id = str(away['teamid'])
    points = home[target_col] - away[target_col]

    # Discount home team's performance.
    teams[home_id][game] = (1.0 - home['is_home'] * .25) * current_discount
    teams[away_id][game] = (-1.0 + away['is_home'] * .25) * current_discount
    result[game] = points

  return pd.DataFrame(teams)

def buildPower(X, y, coerce_fn, acc=0.0001, alpha=1.0):
  y = pd.Series([coerce_fn(val) for val in y])
  model = world_cup.buildModel(y, X, acc=acc, alpha=alpha)

  # print model.summary()
  params = np.exp(model.params)
  del params['intercept']
  params = params[params <> 1.0]
  max_param = params.max()
  min_param = params.min()
  range = max_param - min_param
  if len(params) == 0 or range < 0.0001:
    return None
  
  # return standardizeCol(params).to_dict()
  params = params.sub(min_param)
  params = params.div(range)
  qqs = np.percentile(params, [25, 50, 75])
  def snap(val): 
    for ii in xrange(len(qqs)):
      if (qqs[ii] > val): return ii * 0.33
    return 1.0
    
  # Snap power data to rought percentiles.
  # return params.apply(snap).to_dict()
  # return params.apply(lambda val: 0.0 if val < q1 else (.5 if val < q2 else 1.0)).to_dict()
  return params.to_dict()

def getPowerMap(competition, competition_data, col, coerce_fn):
  power = {}
  acc = 0.000001
  alpha = 0.5
  # Restrict the number of competitions so that we can make
  # sure we'll work with WC data.
  # competition_data = competition_data.iloc[:100]
  while True:
    if alpha < 0.1:
      break;
    try:
      teams = buildTeamMatrix(competition_data, col)
      y = teams[col]
      del teams[col]
      competition_power = buildPower(teams, y, coerce_fn, acc, alpha)
      if competition_power is None:
        alpha /= 2
        print 'Reducing alpha for %s to %f due lack of dynamic range' % (competition, alpha)
      else:
        power.update(competition_power)
        break
    except LinAlgError, err:
      alpha /= 2  
      print 'Reducing alpha for %s to %f due to error %s' % (competition, alpha, err)

  if alpha < 0.1:
    print "Skipping power ranking for competition %s column %s" % (
      competition, col)
    return {}
  return power

def addPower(data, power_train_data, cols):
  data = data.copy()
  competitions = data['competitionid'].unique()
  for (col, coerce_fn, final_name) in cols:
    power = {}
    for competition in competitions:
      competition_data = power_train_data[power_train_data['competitionid'] == competition]
      power.update(getPowerMap(competition, competition_data, col, coerce_fn))

    names = {}
    power_col = pd.Series(np.zeros(len(data)), data.index)
    for index in xrange(len(data)):
      teamid = str(data.iloc[index]['teamid'])
      # if not teamid in power:
      #  print "Missing power data for %s" % teamid
      names[data.iloc[index]['team_name']] = power.get(teamid, 0.5)
      # print "%d: %s -> %s" % (index, teamid, power.get(teamid, 0.5))
      power_col.iloc[index] = power.get(teamid, 0.5)
    print ['%s: %0.03f' % (x[0], x[1]) for x in sorted(names.items(), key=(lambda x: x[1]))]
    data['power_%s' % (final_name)] = power_col
  return data
