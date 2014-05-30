import pandas as pd
import statsmodels.api as sm
import pylab as pl

class ROCData(object):
        """ Class that generates an ROC Curve for the data.
                Data is in the following format: a list l of tutples t
                where:
                        t[0] = 1 for positive class and t[0] = 0 for negative class
                        t[1] = score
                        t[2] = label
        """
        def __init__(self,data,linestyle='rx-'):
                """ Constructor takes the data and the line style for plotting the ROC Curve.
                        Parameters:
                                data: The data a listl of tuples t (l = [t_0,t_1,...t_n]) where:
                                          t[0] = 1 for positive class and 0 for negative class
                                          t[1] = a score
                                          t[2] = any label (optional)
                                lineStyle: THe matplotlib style string for plots.
                                
                        Note: The ROCData is still usable w/o matplotlib. The AUC is still available, 
                              but plots cannot be generated.
                """
                self.data = sorted(data,lambda x,y: cmp(y[1],x[1]))
                self.linestyle = linestyle
                self.auc() #Seed initial points with default full ROC
        
        def auc(self,fpnum=0):
                """ Uses the trapezoidal ruel to calculate the area under the curve. If fpnum is supplied, it will 
                        calculate a partial AUC, up to the number of false positives in fpnum (the partial AUC is scaled
                        to between 0 and 1).
                        It assumes that the positive class is expected to have the higher of the scores (s(+) < s(-))
                        Parameters:
                                fpnum: The cumulativr FP count (fps)
                        Return:
                        
                """
                fps_count = 0
                relevant_pauc = []
                current_index = 0
                max_n = len([x for x in self.data if x[0] == 0])
                if fpnum == 0:
                        relevant_pauc = [x for x in self.data]
                elif fpnum > max_n:
                        fpnum = max_n
                #Find the upper limit of the data that does not exceed n FPs
                else:
                        while fps_count < fpnum:
                                relevant_pauc.append(self.data[current_index])
                                if self.data[current_index][0] == 0:
                                        fps_count += 1
                                current_index +=1
                total_n = len([x for x in relevant_pauc if x[0] == 0])
                total_p = len(relevant_pauc) - total_n
                
                #Convert to points in a ROC
                previous_df = -1000000.0
                current_index = 0
                points = []
                tp_count, fp_count = 0.0 , 0.0
                tpr, fpr = 0, 0
                while current_index < len(relevant_pauc):
                        df = relevant_pauc[current_index][1]
                        if previous_df != df:
                                points.append((fpr,tpr,fp_count))
                        if relevant_pauc[current_index][0] == 0:
                                fp_count +=1
                        elif relevant_pauc[current_index][0] == 1:
                                tp_count +=1
                        fpr = fp_count/total_n
                        tpr = tp_count/total_p
                        previous_df = df
                        current_index +=1
                points.append((fpr,tpr,fp_count)) #Add last point
                points.sort(key=lambda i: (i[0],i[1]))
                self.derived_points = points
                
                return self._trapezoidal_rule(points)


        def _trapezoidal_rule(self,curve_pts):
                """ Method to calculate the area under the ROC curve"""
                cum_area = 0.0
                for ix,x in enumerate(curve_pts[0:-1]):
                        cur_pt = x
                        next_pt = curve_pts[ix+1]
                        cum_area += ((cur_pt[1]+next_pt[1])/2.0) * (next_pt[0]-cur_pt[0])
                return cum_area
                
        
        def plot(self,title='',include_baseline=False,equal_aspect=True):
                """ Method that generates a plot of the ROC curve 
                        Parameters:
                                title: Title of the chart
                                include_baseline: Add the baseline plot line if it's True
                                equal_aspect: Aspects to be equal for all plot
                """
                
                pl.clf()
                pl.plot([x[0] for x in self.derived_points], [y[1] for y in self.derived_points], self.linestyle)
                if include_baseline:
                        pl.plot([0.0,1.0], [0.0,1.0],'k-.')
                pl.ylim((0,1))
                pl.xlim((0,1))
                pl.xticks(pl.arange(0,1.1,.1))
                pl.yticks(pl.arange(0,1.1,.1))
                pl.grid(True)
                if equal_aspect:
                        cax = pl.gca()
                        cax.set_aspect('equal')
                pl.xlabel('1 - Specificity')
                pl.ylabel('Sensitivity')
                pl.title(title)
                
                pl.show()
                
        
        def confusion_matrix(self,threshold,do_print=False):
                """ Returns the confusion matrix (in dictionary form) for a fiven threshold
                        where all elements > threshold are considered 1 , all else 0.
                        Parameters:
                                threshold: threshold to check the decision function
                                do_print:  if it's True show the confusion matrix in the screen
                        Return:
                                the dictionary with the TP, FP, FN, TN
                """
                pos_points = [x for x in self.data if x[1] >= threshold]
                neg_points = [x for x in self.data if x[1] < threshold]
                tp,fp,fn,tn = self._calculate_counts(pos_points,neg_points)
                if do_print:
                        print "\t Actual class"
                        print "\t+(1)\t-(0)"
                        print "+(1)\t%i\t%i\tPredicted" % (tp,fp)
                        print "-(0)\t%i\t%i\tclass" % (fn,tn)
                return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

        def _calculate_counts(self,pos_data,neg_data):
                """ Calculates the number of false positives, true positives, false negatives and true negatives """
                tp_count = len([x for x in pos_data if x[0] == 1])
                fp_count = len([x for x in pos_data if x[0] == 0])
                fn_count = len([x for x in neg_data if x[0] == 1])
                tn_count = len([x for x in neg_data if x[0] == 0])
                return tp_count,fp_count,fn_count, tn_count
