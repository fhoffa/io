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

import features
from roc import ROCData

def teamPredict(all_predictions, cnt):
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
        
def runTeam(data, ignore_cols, target_col):
  (train, test) = split(data, target_col)
  (y_test, X_test) = extract_target(test, target_col)
  (y_train, X_train) = extract_target(train, target_col)
  X_train2 = coerceDf(clone_and_drop(X_train, ignore_cols))    
  X_test2 = coerceDf(clone_and_drop(X_test, ignore_cols))

  models = []
  for (param, test_f) in [(0, check_eq(0)), 
                          (1, check_ge(1)), 
                          (2, check_ge(2)),
                          (3, check_ge(3))
                          # (4, check_ge(4))
                          ]:    
    y = [test_f(yval) for yval in y_train]
    features = featureSelect(y, X_train2)
    models.append((param, test_f, build_model(y, X_train2[features]), features))
    
  count = len(data[target_col])

  all_predictions = []
  for (param, test_f, model, features) in models:
    predictions = model.predict(X_test2[features])
    base_count = sum([test_f(yval) for yval in data[target_col]])
    baseline = base_count * 1.0 / count
    y = [test_f(yval) for yval in y_test]
    validate(param, y, predictions, baseline, compute_auc=True)
    all_predictions.append((param, predictions))
    print '%s: %s: %s' % (target_col, param, model.summary())

  (team_predictions, probs) = teamPredict(all_predictions, len(y_test))
  # print team_predictions
  # print probs
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

# Because we don't have data on both teams during a match, we want
# to drop any match we don't have info about both teams. This can happen
# if we have fewer than 10 previous games from a particular team.
def drop_unbalanced_matches(data):
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

def splice_target_col(col):
  """ Swap the even numered rows """
  for ii in xrange(0, len(col), 2):
    val = col[ii]
    col[ii] -= col[ii + 1]
    col[ii+1] -= val
  return col

def splice_col(col):
  """ Swap the even numered rows """
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
  opp = opp.apply(splice_col)
  del opp['opp_intercept']
  del opp['opp_is_home']
  
  return data.join(opp)

def split(data, target_col):
    train_vec = []
    while len(train_vec) < len(data):
        rnd = random.random()
        train_vec.append(rnd > .4)
        train_vec.append(rnd > .4)
            
    test_vec = [not val for val in train_vec]
    train = data[train_vec]
    test = data[test_vec]
    if len(train) % 2 != 0:
        raise "Unexpected train length"
    if len(test) % 2 != 0:
        raise "Unexpected test length"
    return (train, test)

def extract_target(data, target_col):
    # Todo... use a slice rather than a copy
    y = data[target_col]
    train_df = data.copy()
    del train_df[target_col]
    return y, train_df

def check_ge(n): return lambda (x): int(x) >= int(n)
def check_eq(n): return lambda (x): int(x) == int(n)

def featureSelect(y, X):
  if False:
    logit = sm.Logit(y, X)
    result =  logit.fit(maxiter=1024)
    keep_cols = []
    bse = result.bse
    params = result.params
    for ii in xrange(len(X.columns)):
      param = abs(params[ii])
      err = bse[ii]
      if param - err / 2 > 0:
	keep_cols.append(X.columns[ii])
    return keep_cols
  else:
    print "Features: %s" % (X.columns,) 
    return X.columns

def build_model(y, X):
  logit = sm.Logit(y, X)
  return logit.fit_regularized(maxiter=1024, alpha=2.0)

def classify(probabilities, proportions, levels=None):
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

def validate(k, y, predictions, baseline, compute_auc=False):

  if len(y) <> len(predictions):
    raise Exception("Length mismatch %d vs %d" % (len(y), len(predictions)))
  if baseline > 1.0:
    # Baseline number is expected count, not proportion. Get the proportion.
    baseline = baseline * 1.0 / len(y)

  # print "Validating %d entries" % (len(y),)
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
    first_type = None
    return [1.0 * val for val in vals]

def coerceDf(df): return whiten(df.apply(coerceTypes))

def whiten_col(col):
  std = np.std(col)
  mean = np.mean(col)
  if abs(std) > 0.01:
    return [(val - mean)/std for val in col]
  else:
    return col

  
def whiten(df):
   return df.apply(whiten_col)

def clone_and_drop(data, drop_cols):
    clone = data.copy()
    for col in drop_cols:
        if col in clone.columns:
            del clone[col]
    return clone

def normalize(vec):
    total = float(sum(vec))
    return [val / total for val in vec]

def teamTest(y):
  results = []
  for game in range(len(y)/2):
    g0 = int(y.iloc[game * 2])
    g1 = int(y.iloc[game * 2 + 1])
    if g0 > g1: results.append(3)
    elif g0 == g1: results.append(1)
    else: results.append(0)
  return results

def check_data(data):
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


# Given a vector [win prob, draw prob, loss prob]
# return the 3, 1, or 0 depending on the most likely
# outcome (3 for win, 1 for draw, 0 for loss).
def predictWdl(wdl):
    if wdl[0] > wdl[1] and wdl[0] > wdl[2]:
        return 3
    elif wdl[1] > wdl[2]:
        return 1
    else:
        return 0

def predictWl(wl):
    return (3 if wl[0] >= wl[1] else 0)
    
def teamGamePredict(all_predictions, cnt):
    predictions = []
    for game in range(cnt/2):
        p0 = []
        p1 = []
        for (points, point_predictions) in all_predictions:
            p0.append(point_predictions[game * 2])
            p1.append(point_predictions[game * 2 + 1])
        p0 = normalize(p0)
        p1 = normalize(p1)
        
        pW0 = p0[0]
        pW1 = p1[0]
        
        dW = pW0 - pW1
        
        current = -1
        thresh = 0.1
        if dW > thresh: current = 3
        elif dW < -thresh: current = 0
        else: current = 1    
        
        # print "P0: %s, P1: %s, pred: %d" % (p0, p1, current)
        predictions.append(current) 
        # predictions.append(expect0)
    return predictions

def teamGamePredictNoDraw(all_predictions, cnt):
    predictions = []
    for game in range(cnt/2):
        pW0 = all_predictions[game * 2]
        pW1 = all_predictions[game * 2 + 1]
        
        dW = pW0 - pW1
        
        predictions.append(dW)
        """
        current = -1
        thresh = 0.05
        if dW > thresh: current = 3
        elif dW < -thresh: current = 0
        else: current = 1    
        """
        # expect0 = predictWl(p0)
        # expect1 = predictWl(p1)
        """
        if expect0 == 3 and expect1 == 0:
            current = 3
        elif expect0 == 1 and expect1 == 1:
            current = 1
        elif expect0 == 0 and expect1 == 3:
            current = 0
        else:
            # Difference of opinion.
            current = 1
        """
        # print "P0: %s, P1: %s, pred: %d" % (pW0, pW1, current)
        # predictions.append(current) 
        # predictions.append(expect0)
    return predictions

def runGame(data, ignore_cols, target_col):
  (train, test) = split(data, target_col)
  (y_test, X_test) = extract_target(test, target_col)
  (y_train, X_train) = extract_target(train, target_col)
  X_train2 = coerceDf(clone_and_drop(X_train, ignore_cols))    
  X_test2 = coerceDf(clone_and_drop(X_test, ignore_cols))

  models = []
  for (param, test_f) in [(3, check_eq(3)), 
                          # (1, check_eq(1)), 
                          (0, check_eq(0))
                         ]:
    y = [test_f(yval) for yval in y_train]
    features = featureSelect(y, X_train2)
    models.append((param, test_f, build_model(y, X_train2[features]), features))
    
  count = len(data[target_col])

  all_predictions = []
  for (param, test_f, model, features) in models:
    predictions = model.predict(X_test2[features])
    base_count = sum([test_f(yval) for yval in data[target_col]])
    baseline = base_count * 1.0 / count
    print "Count: %d / Baseline %f" % (base_count, baseline)

    y = [test_f(yval) for yval in y_test]
    # print y
    validate(param, y, predictions, baseline, compute_auc=True)
    all_predictions.append((param, predictions))
    
  team_predictions = teamGamePredict(all_predictions, len(y_test))
  all_team_results = teamTest(data['points'])
  test_team_results = teamTest(y_test)
  zips = zip(team_predictions, test_team_results)
  print (zips)

  validate('w', 
           [pts == 3 for pts in test_team_results], 
           [1.0 if pts == 3 else 0.0 for pts in team_predictions],
           sum([pts == 3 for pts in all_team_results]) * 1.0 / len(all_team_results))
  validate('d', 
           [pts == 1 for pts in test_team_results], 
           [1.0 if pts == 1 else 0.0 for pts in team_predictions],
           sum([pts == 1 for pts in all_team_results]) * 1.0 / len(all_team_results))
  validate('l', 
           [pts == 0 for pts in test_team_results], 
           [1.0 if pts == 0 else 0.0 for pts in team_predictions],
           sum([pts == 0 for pts in all_team_results]) * 1.0 / len(all_team_results))
  correct = sum([int(pred) == int(res) for (pred, res) in zips])
  print "Pct correct = %d/%d=%g" % (correct, len(zips), correct * 1.0 / len(zips))
  print confusion_matrix(test_team_results, team_predictions)

def runGameNoDraw(data, ignore_cols, target_col):
  (train, test) = split(data, target_col)
  train = train.loc[train[target_col] <> 1]
  # test = test.loc[test[target_col] <> 1]
  (y_test, X_test) = extract_target(test, target_col)
  (y_train, X_train) = extract_target(train, target_col)
  X_train2 = splice(coerceDf(clone_and_drop(X_train, ignore_cols))) 
  X_test2 = splice(coerceDf(clone_and_drop(X_test, ignore_cols)))

  (param, test_f) = (3, check_eq(3)) 
  y = [test_f(yval) for yval in y_train]
  features = featureSelect(y, X_train2)
  model = build_model(y, X_train2[features])
  print '%s: %s: %s' % (target_col, param, model.summary())
    
  count = len(data[target_col])

  all_predictions = []
  predictions = model.predict(X_test2[features])
  base_count = sum([test_f(yval) for yval in data[target_col]])
  baseline = base_count * 1.0 / count
  print "Count: %d / Baseline %f" % (base_count, baseline)

  y = [test_f(yval) for yval in y_test]
  # print y
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
  print zip(X_train['team_name'],X_train['op_team_name'], X_train['matchid'], X_train['is_home'],  predictions)
  zips = zip(predicted, test_team_results)
  correct = sum([int(pred) == int(res) for (pred, res) in zips])
  print confusion_matrix(test_team_results, predicted)
  print "Pct correct = %d/%d=%g" % (correct, len(zips), correct * 1.0 / len(zips))


def runSimple(data, ignore_cols):
  target_col = 'points'
  (train, test) = split(data, target_col)
  print  test.describe()
  print test.std()
  (y_test, X_test) = extract_target(test, target_col)
  (y_train, X_train) = extract_target(train, target_col)
  X_train2 = splice(coerceDf(clone_and_drop(X_train, ignore_cols)))   
  X_test2 = splice(coerceDf(clone_and_drop(X_test, ignore_cols)))

  #print X_train2.describe()
  #print X_train2.head()

  y = [check_eq(3)(yval) for yval in y_train]
  features = featureSelect(y, X_train2)
  model = build_model(y, X_train2[features])
    
  count = len(data[target_col])

  all_predictions = []
  predictions = model.predict(X_test2[features])
  base_count = sum([check_eq(3)(yval) for yval in data[target_col]])
  baseline = base_count * 1.0 / count
  y = [check_eq(3)(yval) for yval in y_test]
  validate(3, y, predictions, baseline, compute_auc=True)
  print model.summary()
  # print np.exp(model.params)
  print confusion_matrix(y, [prediction > .50 for prediction in predictions])
  print zip(X_train['team_name'],X_train['op_team_name'], X_train['matchid'], 
            predictions, data[target_col])


def prepare_data(data):
  data = data.copy()
  data['intercept'] = 1.0
  data = drop_unbalanced_matches(data)
  check_data(data)
  return data

