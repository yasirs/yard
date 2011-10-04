"""\
Routines and classes for drawing ROC curves, calculating
sensitivity, specificity, precision, recall, TPR, FPR and such.
"""

from __future__ import division

from bisect import bisect_left
from itertools import izip

from yard.mathematics import rank
from yard.utils import axis_label

__author__  = "Tamas Nepusz"
__email__   = "tamas@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2010, Tamas Nepusz"
__license__ = "MIT"


# pylint: disable-msg=C0103, E0202, E0102, E1101, R0913
class BinaryConfusionMatrix(object):
    """Class representing a binary confusion matrix.

    This class acts like a 2 x 2 matrix, it can also be indexed like
    that, but it also has some attributes to make the code using
    binary confusion matrices easier to read. These attributes are:

      - ``tp``: number of true positives
      - ``tn``: number of true negatives
      - ``fp``: number of false positives
      - ``fn``: number of false negatives
    """

    __slots__ = ("tp", "fp", "tn", "fn")

    def __init__(self, data=None, tp=0, fp=0, fn=0, tn=0):
        self.tp, self.fp, self.fn, self.tn = tp, fp, fn, tn
        if data:
            self.data = data

    @property
    def data(self):
        """Returns the data stored by this confusion matrix"""
        return [[self.tn, self.fn], [self.fp, self.tp]]

    @data.setter
    def data(self, data):
        """Sets the data stored by this confusion matrix"""
        if isinstance(data, BinaryConfusionMatrix):
            self.data = data.data
            return

        if len(data) != 2:
            raise ValueError("confusion matrix must have 2 rows")
        if any(len(row) != 2 for row in data):
            raise ValueError("confusion matrix must have 2 columns")
        (self.tn, self.fn), (self.fp, self.tp) = data

    @axis_label("Accuracy")
    def accuracy(self):
        """Returns the accuracy, i.e. (TP+TN) / (P+N).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=77, fp=77, fn=23, tn=23)
            >>> matrix.accuracy()
            0.5
        """
        num = self.tp + self.tn
        if num == 0:
            return 0
        den = num + self.fp + self.fn
        return num / den

    @axis_label("Error rate")
    def error_rate(self):
        """Returns the error rate, i.e. (FP+FN) / (P+N).

        Example::

            >>> matrix = BinaryConfusionMatrix(tp=77, fp=77, fn=23, tn=23)
            >>> matrix.error_rate()
            0.5
        """
        num = self.fp + self.fn
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("Fraction of data classified negative")
    def fdn(self):
        """Returns the fraction of data classified as negative (FDN)
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdn(), 6)
            0.545
        """
        num = self.fn + self.tn
        den = num + self.fp + self.tp
        return num / den

    @axis_label("Fraction of data classified positive")
    def fdp(self):
        """Returns the fraction of data classified as positive (FDP)
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdp(), 6)
            0.455
        """
        num = self.fp + self.tp
        den = num + self.fn + self.tn
        return num / den

    @axis_label("False discovery rate")
    def fdr(self):
        """Returns the false discovery date (FDR), also known as prediction
        conditioned fallout. It is defined as FP / (TP+FP).
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fdr(), 6)
            0.307692
        """
        return self.fp / (self.fp + self.tp)

    @axis_label("False negative rate")
    def fnr(self):
        """Returns the false negative rate (FNR), i.e. FN / (FN + TP).
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fnr(), 6)
            0.37
        """
        return self.fn / (self.fn + self.tp)

    @axis_label("False positive rate")
    def fpr(self):
        """Returns the false positive rate (FPR), i.e. FP / (FP + TN).
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.fpr(), 6)
            0.28
        """
        return self.fp / (self.fp + self.tn)

    @axis_label("F-score")
    def f_score(self, f=1.0):
        """Returns the F-score.

        The value of `f` controls the weighting between precision and recall
        in the F-score formula. `f` = 1 means that equal importance is attached
        to precision and recall. In general, recall is considered `f` times more
        important than precision.
        """
        sq = float(f*f)
        num = (1 + sq) * self.tp
        return num / (num + sq * self.fn + self.fp)

    @axis_label("Matthews correlation coefficient")
    def mcc(self):
        """Returns the Matthews correlation coefficient (also known as
        phi correlation coefficient)"""
        num = self.tp * self.tn - self.fp * self.fn
        den = (self.tp + self.fp)
        den *= (self.tp + self.fn)
        den *= (self.tn + self.fp)
        den *= (self.tn + self.fn)
        return num / (den ** 0.5)

    @axis_label("Negative predictive value")
    def npv(self):
        """Returns the negative predictive value (NPV), i.e. TN / (TN+FN).
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.npv(), 4)
            0.6606
        """
        return self.tn / (self.tn + self.fn)

    @axis_label("Odds ratio")
    def odds_ratio(self):
        """Returns the odds ratio.
        
        Example::

            >>> matrix = BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72)
            >>> print round(matrix.odds_ratio(), 3)
            4.378
        """
        num = self.tp * self.tn
        den = self.fp * self.fn
        if den == 0:
            return float('nan') if num == 0 else float('inf')
        return num / den

    @axis_label("Precision")
    def precision(self):
        """Returns the precision, a.k.a. the positive predictive value (PPV), i.e.
        TP / (TP+FP)."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 1.0

    @axis_label("Recall")
    def recall(self):
        """Returns the recall, a.k.a. the true positive rate (TPR) or sensitivity,
        i.e. TP / (TP+FN)."""
        return self.tp / (self.tp + self.fn)

    @axis_label("Rate of negative predictions")
    def rnp(self):
        """Returns the rate of negative predictions, i.e. (TN+FN) / (TN+FN+TP+FP)."""
        num = self.tn + self.fn
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("Rate of positive predictions")
    def rpp(self):
        """Returns the rate of positive predictions, i.e. (TP+FP) / (TN+FN+TP+FP)."""
        num = self.tp + self.fp
        if num == 0:
            return 0
        return num / (self.tp + self.fp + self.tn + self.fn)

    @axis_label("True negative rate")
    def tnr(self):
        """Returns the true negative rate (TNR), a.k.a. specificity"""
        return self.tn / (self.fp + self.tn)

    def __eq__(self, other):
        return self.tp == other.tp and self.tn == other.tn and \
               self.fp == other.fp and self.fn == other.fn

    def __getitem__(self, coords):
        obs, exp = coords
        return self._data[obs][exp]

    def __hash__(self):
        return hash((self.tp, self.tn, self.fp, self.fn))

    def __repr__(self):
        return "%s(tp=%d, fp=%d, fn=%d, tn=%d)" % \
                (self.__class__.__name__, self.tp, self.fp, self.fn, self.tn)

    def __setitem__(self, coords, value):
        obs, exp = coords
        self._data[obs][exp] = value

    # Some aliases
    ppv = precision
    sensitivity = recall
    tpr = recall
    specificity = tnr
    fallout = fpr
    miss = fnr
    phi = mcc

    
class BinaryClassifierData(object):
    """Class representing the output of a binary classifier.

    The dataset must contain ``(x, y)`` pairs where `x` is a predicted
    value and `y` defines whether the example is positive or negative.
    When `y` is less than or equal to zero, it is considered a negative
    example, otherwise it is positive. ``False`` also means a negative
    and ``True`` also means a positive example.

    The class has an instance attribute called `title`, representing
    the title of the dataset. This title will be used in ROC curve
    plots in the legend. If the `title` is ``None``, the dataset will
    not appear in legends.
    """

    def __init__(self, data_orScores, title_orLabels=None, *args):
        self._title = None

        if isinstance(data_orScores, BinaryClassifierData):
            # use the given object
            try:
		from numpy import array
                self.scores = array(data_orScores.scores)
		sort_inds = self.scores.argsort()
                self.labels = array(data_orScores.labels)[sort_inds]
                del sort_inds
            except ImportError:
                self.scores = data_orScores.scores
                sort_inds = sorted(xrange(len(self.scores)), key=self.scores.__getitem__)
                self.labels = [data_orScores.labels[i] for i in sort_inds]
                del sort_inds
        else:
            # we either have data as pairs or two separate lists
            if ((title_orLabels is not None) and (len(data_orScores)==len(title_orLabels)) and (len(args)==0)):
                # data as two separate lists
                try:
                    from numpy import array
                    self.scores = array(data_orScores)
		    sort_inds = self.scores.argsort()
                    self.scores = self.scores[sort_inds]
                    self.labels = array(title_orLabels)>0
                    self.labels = self.labels[sort_inds]
                    del sort_inds
                except ImportError:
                    self.scores = data_orScores
                    sort_inds = sorted(xrange(len(self.scores)), key=self.scores.__getitem__)
                    self.scores = [self.scores[i] for i in sort_inds]
                    self.labels = [title_orLabels[i]>0 for i in sort_inds]
                    del sort_inds
                if len(args):
                    self.title = args[0]
            else:
                # data as pairs
                try:
                    from numpy import array
                    self.scores = array([x[0] for x in data_orScores])
                    sort_inds = self.scores.argsort()
                    self.scores = self.scores[sort_inds]
                    self.labels = array([x[1] for x in data_orScores])>0
                    self.labels = self.labels[sort_inds]
                    del sort_inds
                except ImportError:
                    self.scores, self.labels = zip(*data_orScores)
                    sort_inds = sorted(xrange(len(self.scores)), key=self.scores.__getitem__)
                    self.scores = [self.scores[i] for i in sort_inds]
                    self.labels = [data_orScores[i][1]>0 for i in sort_inds]
                    del sort_inds
                self.title = title_orLabels
        try:
            self.total_positives = self.labels.sum()
        except AttributeError:
            self.total_positives = sum(self.labels)
        self.total_points = len(self.scores)
        self.total_negatives = self.total_points - self.total_positives
        try:
            from numpy import cumsum
            self.cumulative_positives = cumsum(self.labels)
        except ImportError:
            def cumsum(seq):
                s = 0
                for n in seq:
                    s += n
                    yield s
            self.cumulative_positives = list(cumsum(self.labels))

    def __getitem__(self, index):
        return tuple(self.scores[index],self.labels[index])

    def __len__(self):
        return len(self.scores)

    @axis_label("True Positive Rate")
    def tpr(self):
        try:
             from numpy import linspace
             return self.cumulative_positives/linspace(1,self.total_points,self.total_points)
        except ImportError:
             def tprgen(seq):
                 i = 1.0
                 for n in seq:
                     yield n/i
                     i += 1
             return list(tprgen(self.cumulative_positives))

    @axis_label("False Positive Rate")
    def fpr(self):
        try:
             from numpy import linspace
             return 1 - (self.cumulative_positives)/linspace(1,self.total_points,self.total_points)
        except ImportError:
             def fprgen(seq):
                 i = 1.0
                 for n in seq:
                     yield (i-n)/i
                     i += 1
             return list(fprgen(self.cumulative_positives))
      
    @axis_label("Precision")
    def precision(self):
        try:
             from numpy import linspace
             return self.cumulative_positives/linspace(1,self.total_points,self.total_points)
        except ImportError:
             def tprgen(seq):
                 i = 1.0
                 for n in seq:
                     yield n/i
                     i += 1
             return list(tprgen(self.cumulative_positives))

    @axis_label("Recall")
    def recall(self):
        try:
             from numpy import linspace
             return self.cumulative_positives/self.total_positives
        except ImportError:
             return list((n+0.0)/self.total_positives for n in self.cumulative_positives)


    @staticmethod
    def _normalize_point(point):
        """Normalizes a data point by setting the second element
        (which tells whether the example is positive or negative)
        to either ``True`` or ``False``.
        
        Returns the new data point as a tuple."""
        return point[0], point[1] > 0

    def get_confusion_matrix(self, threshold):
        """Returns the confusion matrix at a given threshold
        
        Example::
            
            >>> outcomes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            >>> expected = [0, 0, 0, 1, 0, 1, 1, 1, 1]
            >>> data = BinaryClassifierData(zip(outcomes, expected))
            >>> data.get_confusion_matrix(0.2)
            BinaryConfusionMatrix(tp=5, fp=3, fn=0, tn=1)
            >>> data.get_confusion_matrix(0.75)
            BinaryConfusionMatrix(tp=2, fp=0, fn=3, tn=4)
        """
        result = [[0, 0], [0, 0]]
        # Find the index in the data where the predictions start to
        # exceed the threshold
        idx = bisect_left(self.scores, threshold)
        if idx <= len(self.scores) // 2:
            for is_pos in self.labels[:idx]:
                result[0][is_pos] += 1 # this can probably be written without an explicit loop TODO
            result[1][0] = self.total_negatives - result[0][0]
            result[1][1] = self.total_positives - result[0][1]
        else:
            for is_pos in self.labels[idx:]:
                result[1][is_pos] += 1
            result[0][0] = self.total_negatives - result[1][0]
            result[0][1] = self.total_positives - result[1][1]
        return BinaryConfusionMatrix(data=result)

    def get_negative_ranks(self):
        """Returns the ranks of the negative instances."""
        ranks = rank(self.scores)
        return [ranks[idx] for idx, truth in enumerate(self.labels) if not truth] # TODO can probably be written better with scipy

    def get_positive_ranks(self):
        """Returns the ranks of the positive instances."""
        ranks = rank(self.scores)
        return [ranks[idx] for idx, truth in enumerate(self.labels) if truth]

    def iter_confusion_matrices(self, thresholds=None):
        """Iterates over the possible prediction thresholds in the
        dataset and yields tuples containing the threshold and the
        corresponding confusion matrix. This method can be used to
        generate ROC curves and it is more efficient than getting
        the confusion matrices one by one.
        
        @param thresholds: the thresholds for which we evaluate the
          confusion matrix. If it is ``None``, all possible thresholds
          from the dataset will be evaluated. If it is an integer `n`,
          we will choose `n` threshold levels equidistantly from
          the range `0-1`. If it is an iterable, then each member
          yielded by the iterable must be a threshold."""
        if not len(self):
            return

        if thresholds is None:
            try:
                from numpy import append
                thresholds = append(self.scores,float('inf'))
            except ImportError:
                import copy
                thresholds = copy.copy(self.scores)
                thresholds.append(float('inf'))
        elif not hasattr(thresholds, "__iter__"):
            n = float(thresholds)
            thresholds = [i/n for i in xrange(thresholds)]
        try:
            import numpy
            thresholds = numpy.unique(thresholds)
        except ImportError:
            thresholds = sorted(set(thresholds))

        if len(thresholds)==0:
            return

        result = BinaryConfusionMatrix(tp=self.total_positives,fp=self.total_negatives,tn=0,fn=0)

        idx, n = 0, len(self)
        for threshold in thresholds:
            while idx < n:
                score = self.scores[idx]
                if score >= threshold:
                    break
                if self.labels[idx]:
                    # This data point is a positive example. Since
                    # we are below the threshold now (and we weren't
                    # in the previous iteration), we have one less
                    # TP and one more FN
                    result.tp -= 1
                    result.fn += 1
                else:
                    # This data point is a negative example. Since
                    # we are below the threshold now (and we weren't
                    # in the previous iteration), we have one more
                    # TN and one less FP
                    result.tn += 1
                    result.fp -= 1
                idx += 1
            yield threshold, BinaryConfusionMatrix(result)
    
    @property
    def title(self):
        """The title of the plot"""
        return self._title

    @title.setter
    def title(self, value):
        """Sets the title of the plot"""
        if value is None or isinstance(value, (str, unicode)):
            self._title = value
        else:
            self._title = str(value)


