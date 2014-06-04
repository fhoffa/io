"""
    Predicts soccer outcomes using logistic regression.
    How to run:
import features

# Read the features from bigquery.
data = features.get_features()

not_train_cols = features.get_non_feature_columns() 

# There are three different ways of running the prediction. 
# The simplest is:
world_cup.runSimple(data, not_train_cols)

# The best (currently) is:
world_cup.runGameNoDraw(data, not_train_cols)

# world_cup.runTeam(data, not_train_cols)
"""

import numpy as np
import pandas as pd
from pandas.io import gbq
import pylab as pl
import random
import scipy.cluster
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

from roc import ROCData

def teamPredict(all_predictions, cnt):
    """ Given an list of arrays of predictions, where each array is the prediction
        of a single goal count (goals > k), predict which team will win. 
    """
    predictions = []
    probs = []
    for game in range(cnt/2):
        p0 = []
        p1 = []
        for (goals, goal_predictions) in all_predictions:
            p0.append(goal_predictions[game * 2])
            p1.append(goal_predictions[game * 2 + 1])
        # Add extra since we only have 3 entries instead of 4
        p0.append(0.0)
        p1.append(0.0)
        
        p0 = normalize(p0)
        p1 = normalize(p1)
        pDraw = (
            p0[0] * p1[0] + 
            p0[1] * p1[1] +
            p0[2] * p1[2] +
            p0[3] * p1[3] + 
            p0[4] * p1[4]
            )
        pWin = (
            p1[0] * (p0[1] + p0[2] + p0[3] + p0[4]) +
            p1[1] * (p0[2] + p0[3] + p0[4]) +
            p1[2] * (p0[3] + p0[4]) +
            p1[3] * (p0[4])
        )
        pLose = 1.0 - pDraw - pWin
        probs.append((pWin, pLose, pDraw))
        if pWin >= pDraw and pWin >= pLose:
          predictions.append(3)
        elif pDraw >= pWin and pDraw >= pLose:
          predictions.append(1)
        else:
           predictions.append(0)
    return (predictions,probs)
        
# This model doesn't seem to work as well as the runGameNoDraw version.
# Use it instead.
def runTeam(data, ignore_cols, target_col='goals'):
  """ Runs a goal-based prediciton that predicts the probability
      distribution for goals scored by each team, then predicts the
      winner based on this. """
  data = prepareData(data.copy())
  (train, test) = split(data, target_col)
  (y_test, X_test) = extractTarget(test, target_col)
  (y_train, X_train) = extractTarget(train, target_col)
  X_train2 = splice(coerceDf(cloneAndDrop(X_train, ignore_cols)))    
  X_test2 = splice(coerceDf(cloneAndDrop(X_test, ignore_cols)))

  models = []
  for (param, test_f) in [(0, check_eq(0)), 
                          (1, check_ge(1)), 
                          (2, check_ge(2)),
                          (3, check_ge(3))
                          # (4, check_ge(4))
                          ]:    
    y = [test_f(yval) for yval in y_train]
    features = X_train2.columns
    models.append((param, test_f, buildModel(y, X_train2[features]), features))
    
  count = len(data[target_col])

  all_predictions = []
  for (param, test_f, model, features) in models:
    predictions = predictModel(model, X_test2[features])
    base_count = sum([test_f(yval) for yval in data[target_col]])
    baseline = base_count * 1.0 / count
    y = [test_f(yval) for yval in y_test]
    validate(param, y, predictions, baseline, compute_auc=True)
    all_predictions.append((param, predictions))
    print '%s: %s: %s' % (target_col, param, model.summary())

  (team_predictions, probs) = teamPredict(all_predictions, len(y_test))
  all_team_results = teamTest(data[target_col])
  test_team_results = teamTest(y_test)

  validate('w', 
           [int(pts) == 3 for pts in test_team_results], 
           [1.0 if pts == 3 else 0.0 for pts in team_predictions],
           sum([pts == 3 for pts in all_team_results]) * 1.0 / len(all_team_results))
  validate('d', 
           [int(pts) == 1 for pts in test_team_results], 
           [1.0 if int(pts) == 1 else 0.0 for pts in team_predictions],
           sum([int(pts) == 1 for pts in all_team_results]) * 1.0 / len(all_team_results))
  validate('l', 
           [int(pts) == 0 for pts in test_team_results], 
           [1.0 if int(pts) == 0 else 0.0 for pts in team_predictions],
           sum([int(pts) == 0 for pts in all_team_results]) * 1.0 / len(all_team_results))
  zips = zip(team_predictions, test_team_results)
  correct = sum([int(pred) == int(res) for (pred, res) in zips])
  print "Pct correct = %d/%d=%g" % (correct, len(zips), correct * 1.0 / len(zips))
  print confusion_matrix(test_team_results, team_predictions)

def dropUnbalancedMatches(data):
  """  Because we don't have data on both teams during a match, we want
       to drop any match we don't have info about both teams. This can happen
       if we have fewer than 10 previous games from a particular team.
  """

  keep = []
  i = 0;
  while i < len(data) - 1:
    if data['matchid'][i] == data['matchid'][i+1]:
      keep.append(True)
      keep.append(True)
      i += 2
    else:
      keep.append(False)
      i += 1
  while len(keep) < len(data): keep.append(False)
  return data[keep]

def swapPairwise(col):
  """ Swap rows pairwise; i.e. swap row 0 and 1, 2 and 3, etc.  """
  col = pd.np.array(col)
  for ii in xrange(0, len(col), 2):
    val = col[ii]
    col[ii] = col[ii + 1]
    col[ii+1] = val
  return col

def splice(data):
  """ Splice both rows representing a game into a single one. """
  data = data.copy()
  opp = data.copy()
  opp_cols = ['opp_%s' % (col,) for col in opp.columns]
  opp.columns = opp_cols
  opp = opp.apply(swapPairwise)
  del opp['opp_is_home']
  
  return data.join(opp)

def split(data, target_col, test_proportion=0.4):
  """ Splits a dataframe into a training set and a test set.
      Must be careful because back-to-back rows are expeted to
      represent the same game, so they both must go in the 
      test set or both in the training set.
  """
  
  train_vec = []
  while len(train_vec) < len(data):
    rnd = random.random()
    train_vec.append(rnd > test_proportion) 
    train_vec.append(rnd > test_proportion)
          
  test_vec = [not val for val in train_vec]
  train = data[train_vec]
  test = data[test_vec]
  if len(train) % 2 != 0:
    raise "Unexpected train length"
  if len(test) % 2 != 0:
    raise "Unexpected test length"
  return (train, test)

def extractTarget(data, target_col):
  """ Removes the target column from a data frame, returns the target col
      and a new data frame minus the target. """
  y = data[target_col]
  train_df = data.copy()
  del train_df[target_col]
  return y, train_df

def check_ge(n): return lambda (x): int(x) >= int(n)
def check_eq(n): return lambda (x): int(x) == int(n)

def buildModel(y, X):
  X = X.copy()
  X['intercept'] = 1.0
  logit = sm.Logit(y, X)
  return logit.fit_regularized(maxiter=1024, alpha=4.0)

def classify(probabilities, proportions, levels=None):
  """ Given predicted probabilities and a vector of proportions,
      assign the samples to categories (defined in the levels vector,
      or True/False if a levels vector is not provided). The proportions
      vector decides how many of each category we expect (we'll use
      the most likely values)
  """

  if not levels: levels = [False, True]
  zipped = zip(probabilities, range(len(probabilities)))
  zipped = sorted(zipped, key=lambda tup: tup[0])
  predictions = []
  label_index = 0
  split_indexes = []
  split_start = 0.0
  proportions = normalize(proportions)
  for proportion in proportions:
    split_start += proportion * len(probabilities)
    split_indexes.append(split_start)

  for i in xrange(len(zipped)):
    (prob, initial_index) = zipped[i]
    while i > split_indexes[label_index]: label_index += 1
    predicted = levels[label_index]
    predictions.append((prob, predicted, initial_index))
  
  predictions.sort(key=lambda tup: tup[2]) 
  _, results, _ = zip(*predictions)
  return results

def validate(k, y, predictions, baseline=0.5, compute_auc=False):
  """ Validates binary predictions, computes confusion matrix and AUC.

    Given a vector of predictions and actual values, scores how well we
    did on a prediction. 

    Args:
      k: label of what we're validating
      y: vector of actual results
      predictions: predicted results. May be a probability vector,
        in which case we'll sort it and take the most confident values
        where baseline is the proportion that we want to take as True
        predictions. If a prediction is 1.0 or 0.0, however, we'll take
        it to be a true or false prediction, respectively.
      compute_auc: If true, will compute the AUC for the predictions. 
        If this is true, predictions must be a probability vector.
  """

  if len(y) <> len(predictions):
    raise Exception("Length mismatch %d vs %d" % (len(y), len(predictions)))
  if baseline > 1.0:
    # Baseline number is expected count, not proportion. Get the proportion.
    baseline = baseline * 1.0 / len(y)

  zipped = zip(y, predictions)
  zipped = sorted(zipped, key=lambda tup: -tup[1])
  expect = len(y) * baseline
  
  (tp, tn, fp, fn) = (0, 0, 0, 0)
  for i in xrange(len(y)):
    (yval, prob) = zipped[i]
    if float(prob) == 0.0:
      predicted = False
    elif float(prob) == 1.0:
      predicted = True
    else:
      predicted = i < expect
    if predicted:
        if yval:
            tp += 1
        else:
            fp += 1 
    else:
        if yval:
            fn += 1
        else:
            tn += 1

  p = tp + fn
  n = tn + fp
  # P(1 | predicted(1)) and P(0 | predicted(f))
  pred_t = tp + fp
  pred_f = tn + fn
  p1_t = tp * 1.0 / pred_t if pred_t > 0.0 else -1.0
  p0_f = tn * 1.0 / pred_f if pred_f > 0.0 else -1.0
            
  # Lift = P(1 | t) / P(1)
  p1 = p * 1.0 / (p + n)
  lift = p1_t / p1 if p1 > 0 else 0.0
            
  accuracy = (tp + tn) * 1.0 / len(y)
            
  if compute_auc:
    tup_data = [(1 if y[i] else 0, predictions[i]) for i in range(len(y))]
    auc = ROCData(tup_data).auc()
  else:
    auc = "NA"
  if fp + fn + tp + tn <> len(y):
    raise Exception("Unexpected confusion matrix")

  print "(%s) Base: %g Acc: %g P(1|t): %g P(0|f): %g Lift: %g Auc: %s" % (
    k, baseline, accuracy, p1_t, p0_f, lift, auc)
 
  print "Fp/Fn/Tp/Tn p/n/c: %d/%d/%d/%d %d/%d/%d" % (
    fp, fn, tp, tn, p, n, len(y))
  # roc_data.plot()
  
def coerceTypes(vals):
  """ Makes sure all of the values in a list are floats. """
  first_type = None
  return [1.0 * val for val in vals]

def coerceDf(df): 
  """ Coerces a dataframe to all floats, and whitens the values. """
  return whiten(df.apply(coerceTypes))

def whitenCol(col):
  """ Whitens a single column (subtracts mean and divides by std dev. """
  std = np.std(col)
  mean = np.mean(col)
  if abs(std) > 0.001:
    return [(val - mean)/std for val in col]
  else:
    return col

def whiten(df):
   """ Whitens a dataframe. All fields must be numeric. """
   return df.apply(whitenCol)

def cloneAndDrop(data, drop_cols):
  """ Returns a copy of a dataframe that doesn't have certain columns. """
  clone = data.copy()
  for col in drop_cols:
    if col in clone.columns:
      del clone[col]
  return clone

def normalize(vec):
    total = float(sum(vec))
    return [val / total for val in vec]

def teamTest(y):
  """ Given a vector containing the number of goals scored in a game
      where y[k] (where k % 2 = 0) is the number of goals scored by
      the home team and y[k+1] is the number of goals scored by the
      away team, return a vector of length (len(y) / 2) that returns
      the number of points (3 for win, 1 for draw, 0 for loss) that
      the home team (the kth value) gets.
  """ 

  results = []
  for game in range(len(y)/2):
    g0 = int(y.iloc[game * 2])
    g1 = int(y.iloc[game * 2 + 1])
    if g0 > g1: results.append(3)
    elif g0 == g1: results.append(1)
    else: results.append(0)
  return results

def checkData(data):
  """ Walks a dataframe and make sure that all is well. """ 
  i = 0
  if len(data) % 2 != 0:
      raise Exception("Unexpeted length")
  matches = data['matchid']
  teams = data['teamid']
  op_teams = data['op_teamid']
  while i < len(data) - 1:
    if matches.iloc[i] != matches.iloc[i + 1]:
      raise Exception("Match mismatch: %s vs %s " % (
                      matches.iloc[i], matches.iloc[i + 1]))
    if teams.iloc[i] != op_teams.iloc[i + 1]:
      raise Exception("Team mismatch: match %s team %s vs %s" % (
                      matches.iloc[i], teams.iloc[i], 
                      op_teams.iloc[i + 1]))
    if teams.iloc[i + 1] != op_teams.iloc[i]:
      raise Exception("Team mismatch: match %s team %s vs %s" % (
                      matches.iloc[i], teams.iloc[i + 1], 
                      op_teams.iloc[i]))
    i += 2

def teamGamePredictNoDraw(all_predictions, cnt):
  """ Given a vector of predictions where the 
      kth prediction is the home team an the  k+1th prediction is
      the away team (where k % 2 == 0), return a vector of
      predictions containing the difference between home team win
      probability and away team win probability.
  """
        
  predictions = []
  for game in range(cnt/2):
    pW0 = all_predictions[game * 2]
    pW1 = all_predictions[game * 2 + 1]
    dW = pW0 - pW1
    predictions.append(dW)
  return predictions

def runGameNoDraw(data, ignore_cols, target_col='points'):
  """ Builds and tests a model that:
      1. Given an input dataframe that has:
         <<Game a, Home team, Features>,
          <Game a, Away team, Features>,
          <Game b, Home team, Features>,
          <Game b, Away team, Features>>
         Copies features from Away team to home team's records with a mangled name,
         and copies features from Home team to Away team's records. This allows
         prediction based on statistics about both teams.
          
      2. Builds a model predicting outcome (win, loss) over games that
         did not end in draw. These are the 'strong signal' games.
      3. Builds outcome by taking the difference in probability that the
         home team will win and the probability that the away team will win
         and maps those probabilities to outcomes based on prior probabilites
         for win, loss, draw. That is, if we know that 50% of games are won by
         the home team, 20% are draws, and 30 are wone by the away team, we'll
         mark our most certain 50% as wins, the next 20% as draws, and the
         final 30% as losses.      
  """
  
  data = prepareData(data)
  (train, test) = split(data, target_col)
  # Drop draws in the training set; they're not strong signals, so
  # we don't want to train on them.
  train = train.loc[train[target_col] <> 1]

  (y_test, X_test) = extractTarget(test, target_col)
  (y_train, X_train) = extractTarget(train, target_col)
  X_train2 = splice(coerceDf(cloneAndDrop(X_train, ignore_cols))) 
  X_test2 = splice(coerceDf(cloneAndDrop(X_test, ignore_cols)))

  (param, test_f) = (3, check_eq(3)) 
  y = [test_f(yval) for yval in y_train]
  features = X_train2.columns
  model = buildModel(y, X_train2[features])
  print '%s: %s: %s' % (target_col, param, model.summary())
    
  count = len(data[target_col])

  all_predictions = []
  predictions = predictModel(model, X_test2[features])
  base_count = sum([test_f(yval) for yval in data[target_col]])
  baseline = base_count * 1.0 / count
  print "Count: %d / Baseline %f" % (base_count, baseline)

  y = [test_f(yval) for yval in y_test]
  validate(param, y, predictions, baseline, compute_auc=True)
    
  probabilities = teamGamePredictNoDraw(predictions, len(y_test))
  all_team_results = teamTest(data['points'])
  test_team_results = teamTest(y_test)

  lose_count = sum([pts == 0 for pts in all_team_results])
  draw_count = sum([pts == 1 for pts in all_team_results])
  win_count = sum([pts == 3 for pts in all_team_results])
  predicted = classify(probabilities, [lose_count, draw_count, win_count],
    [0, 1, 3])
  validate('w', 
           [pts == 3 for pts in test_team_results], 
           [1.0 if cl == 3 else 0.0 for cl in predicted],
            win_count)
  validate('d', 
           [int(pts) == 1 for pts in test_team_results], 
           [1.0 if cl == 1 else 0.0 for cl in predicted],
           draw_count)
  validate('l', 
           [int(pts) == 0 for pts in test_team_results], 
           [1.0 if cl == 0 else 0.0 for cl in predicted],
           lose_count)
  print "W/L/D %d/%d/%s" % (win_count, lose_count, draw_count)
  print zip(X_train['team_name'],X_train['op_team_name'],
            X_train['matchid'], predicted, test_team_results)
  zips = zip(predicted, test_team_results)
  correct = sum([int(pred) == int(res) for (pred, res) in zips])
  print confusion_matrix(test_team_results, predicted)
  print "Pct correct = %d/%d=%g" % (
      correct, len(zips), correct * 1.0 / len(zips))

def predictModel(model, X_test):
  X_test = X_test.copy()
  X_test['intercept'] = 1.0
  return model.predict(X_test)

def runSimple(data, ignore_cols):
  """ Runs a simple predictor that will predict if we expect a team to win. """
    
  data = prepareData(data)
  target_col = 'points'
  (train, test) = split(data, target_col)
  (y_test, X_test) = extractTarget(test, target_col)
  (y_train, X_train) = extractTarget(train, target_col)
  X_train2 = splice(coerceDf(cloneAndDrop(X_train, ignore_cols)))   
  X_test2 = splice(coerceDf(cloneAndDrop(X_test, ignore_cols)))

  y = [check_eq(3)(yval) for yval in y_train]
  features = X_train2.columns
  model = buildModel(y, X_train2[features])
    
  count = len(data[target_col])

  all_predictions = []
  predictions = predictModel(model, X_test2[features])
  base_count = sum([check_eq(3)(yval) for yval in data[target_col]])
  baseline = base_count * 1.0 / count
  y = [check_eq(3)(yval) for yval in y_test]
  validate(3, y, predictions, baseline, compute_auc=True)
  print model.summary()
  grounded_predictions = [prediction > 0.50 for prediction in predictions]
  print confusion_matrix(y, grounded_predictions)
  print zip(X_train['team_name'],X_train['op_team_name'], X_train['matchid'], 
            grounded_predictions, data[target_col])

def prepareData(data):
  """ Drops all matches where we don't have data for both teams. """
  
  data = data.copy()
  data = dropUnbalancedMatches(data)
  checkData(data)
  return data

