"""
    Predicts soccer outcomes using logistic regression.
"""

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
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

def _dropUnbalancedMatches(data):
  """  Because we don't have data on both teams during a match, we want
       to drop any match we don't have info about both teams. This can happen
       if we have fewer than 10 previous games from a particular team.
  """

  keep = []
  i = 0
  data = data.dropna()
  while i < len(data) - 1:
    row = data.iloc[i]
    skipped = False
    for col in data:
      if isinstance(col, float) and math.isnan(col):
        keep.append(False)
        i += 1
        skipped = True
       
    if skipped: pass
    elif data.iloc[i]['matchid'] == data.iloc[i+1]['matchid']:
      keep.append(True)
      keep.append(True)
      i += 2
    else:
      keep.append(False)
      i += 1
  while len(keep) < len(data): keep.append(False)
  results = data[keep]
  if len(results) % 2 <> 0:
    raise Exception("Unexpected results")
  return results


def _swapPairwise(col):
  """ Swap rows pairwise; i.e. swap row 0 and 1, 2 and 3, etc.  """
  col = pd.np.array(col)
  for ii in xrange(0, len(col), 2):
    val = col[ii]
    col[ii] = col[ii + 1]
    col[ii+1] = val
  return col

def _splice(data):
  """ Splice both rows representing a game into a single one. """
  data = data.copy()
  opp = data.copy()
  opp_cols = ['opp_%s' % (col,) for col in opp.columns]
  opp.columns = opp_cols
  opp = opp.apply(_swapPairwise)
  del opp['opp_is_home']
  
  return data.join(opp)

def split(data, test_proportion=0.4):
  """ Splits a dataframe into a training set and a test set.
      Must be careful because back-to-back rows are expeted to
      represent the same game, so they both must go in the 
      test set or both in the training set.
  """
  
  train_vec = []
  if len(data) % 2 != 0:
    raise Exception("Unexpected data length")
  while len(train_vec) < len(data):
    rnd = random.random()
    train_vec.append(rnd > test_proportion) 
    train_vec.append(rnd > test_proportion)
          
  test_vec = [not val for val in train_vec]
  train = data[train_vec]
  test = data[test_vec]
  if len(train) % 2 != 0:
    raise Exception("Unexpected train length")
  if len(test) % 2 != 0:
    raise Exception("Unexpected test length")
  return (train, test)

def _extractTarget(data, target_col):
  """ Removes the target column from a data frame, returns the target col
      and a new data frame minus the target. """
  y = data[target_col]
  train_df = data.copy()
  del train_df[target_col]
  return y, train_df

def _check_ge(n): return lambda (x): int(x) >= int(n)
def _check_eq(n): return lambda (x): int(x) == int(n)

def buildModelPoisson(y, X, acc=0.0000001):
  X = X.copy()
  X['intercept'] = 1.0
  logit = sm.Poisson(y, X)
  return logit.fit_regularized(maxiter=10240, alpha=4.0, acc=acc)

l1_alpha = 16.0
def buildModel(y, X, acc=0.00000001, alpha=l1_alpha):
  X = X.copy()
  X['intercept'] = 1.0
  logit = sm.Logit(y, X, disp=False)
  return logit.fit_regularized(maxiter=1024, alpha=alpha, acc=acc, disp=False)

def buildModelMn(y, X, acc=0.0000001, alpha=l1_alpha):
  X = X.copy()
  X['intercept'] = 1.0
  logit = sm.MNLogit(y, X)
  return logit.fit_regularized(maxiter=10240, alpha=alpha, acc=acc, disp=False)


def validate(k, y, predictions, baseline=0.5, compute_auc=False, quiet=True):
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
    y_bool =  [True if yval else False for (yval,_) in zipped]
    x = [xval for (_, xval) in zipped]
    auc_value = roc_auc_score(y_bool, x)
    fpr, tpr, thresholds = roc_curve(y_bool, x)
    pl.plot(fpr, tpr, lw=1.5, label='ROC %s (area = %0.2f)' % (k, auc_value,))
    pl.xlabel('False Positive Rate', fontsize=18)
    pl.ylabel('True Positive Rate', fontsize=18)
    pl.title('ROC curve', fontsize=18)
    auc_value = '%0.03g' % auc_value
  else:
    auc_value = "NA"
  if fp + fn + tp + tn <> len(y):
    raise Exception("Unexpected confusion matrix")

  if quiet:
    print "(%s) Lift: %0.03g Auc: %s" % (k, lift, auc_value)
  else:
    print "(%s) Base: %0.03g Acc: %0.03g P(1|t): %0.03g P(0|f): %0.03g\nLift: %0.03g Auc: %s" % (
      k, baseline, accuracy, p1_t, p0_f, lift, auc_value)
    print "Fp/Fn/Tp/Tn p/n/c: %d/%d/%d/%d %d/%d/%d" % (
      fp, fn, tp, tn, p, n, len(y))
  # roc_data.plot()
  
def _coerceTypes(vals):
  """ Makes sure all of the values in a list are floats. """
  first_type = None
  return [1.0 * val for val in vals]

def _coerceDf(df): 
  """ Coerces a dataframe to all floats, and standardizes the values. """
  return _standardize(df.apply(_coerceTypes))

def _standardizeCol(col):
  """ Standardizes a single column (subtracts mean and divides by std dev). """
  std = np.std(col)
  mean = np.mean(col)
  if abs(std) > 0.001:
    return col.apply(lambda val: (val - mean)/std)
  else:
    return col

def _standardize(df):
   """ Standardizes a dataframe. All fields must be numeric. """
   return df.apply(_standardizeCol)

def _cloneAndDrop(data, drop_cols):
  """ Returns a copy of a dataframe that doesn't have certain columns. """
  clone = data.copy()
  for col in drop_cols:
    if col in clone.columns:
      del clone[col]
  return clone

def _normalize(vec):
    total = float(sum(vec))
    return [val / total for val in vec]

def _games(df):
  """ Drops odd numbered rows in a column. This is used when we
      have two rows representing a game, and we only need 1. """
  return df[[idx % 2 == 0 for idx in xrange(len(df))]] 
  
def _teamTestProb(y):
  results = []
  for game in range(len(y)/2):
    g0 = float(y.iloc[game * 2])
    g1 = float(y.iloc[game * 2 + 1])
    results.append(g0/(g0+g1))
  return results


def extractPredictions(data, predictions):
  probs = _teamTestProb(predictions)
  team0 = []
  team1 = []
  points = []
  for game in xrange(len(data)/2):
    if data['matchid'].iloc[game * 2] <> data['matchid'].iloc[game * 2 + 1]:
      raise Exception("Unexpeted match id %d vs %d", (
                         data['matchid'].iloc[game * 2],
                         data['matchid'].iloc[game * 2 + 1]))
    t0 = data['team_name'].iloc[game * 2]
    t1 = data['op_team_name'].iloc[game * 2]
    if 'points' in data.columns: 
      points.append(data['points'].iloc[game * 2])
    team0.append(t0)
    team1.append(t1)
  df =  pd.DataFrame({'team_name': pd.Series(team0), 
                      'op_team_name': pd.Series(team1),
                      'predicted': pd.Series(probs).mul(100)},
                      columns = ['team_name', 'op_team_name', 'predicted'])
  if len(points) > 0:
    df['points'] = pd.Series(points)
  return df

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


def prepareData(data):
  """ Drops all matches where we don't have data for both teams. """
  
  data = data.copy()
  data = _dropUnbalancedMatches(data)
  checkData(data)
  return data


def _predictModel(model, X_test):
  X_test = X_test.copy().dropna()
  X_test['intercept'] = 1.0
  return model.predict(X_test)


def trainModel(data, ignore_cols):
  # Validate the data
  data = prepareData(data)
  target_col = 'points'
  (train, test) = split(data)
  (y_test, X_test) = _extractTarget(test, target_col)
  (y_train, X_train) = _extractTarget(train, target_col)
  X_train2 = _splice(_coerceDf(_cloneAndDrop(X_train, ignore_cols)))

  y = [int(yval) == 3 for yval in y_train]
  model = buildModel(y, X_train2, alpha=8.0)
  return (model, test)


def predictModel(model, test, ignore_cols):
  """ Runs a simple predictor that will predict if we expect a team to win. """
    
  X_test2 = _splice(_coerceDf(_cloneAndDrop(test, ignore_cols)))
  X_test2['intercept'] = 1.0
  predicted =  model.predict(X_test2)
  result = test.copy()
  result['predicted'] = predicted
  return result

