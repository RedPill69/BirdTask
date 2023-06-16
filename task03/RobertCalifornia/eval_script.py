#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates predictions for the MLPC2023 project.

Author: Jan SchlÃ¼ter
"""

import sys
import os
from argparse import ArgumentParser

import numpy as np
import scipy
import sklearn.metrics

# species order: other, comcuc, cowpig1, eucdov, eueowl1, grswoo, tawowl1
# entry i,j gives the revenue for predicting j when it is labeled i
REVENUES = np.asarray(
        [[ 0.05, -0.2 , -0.2 , -0.2 , -0.2 , -0.2 , -0.2 ],
         [-0.25,  1.  , -0.3 , -0.1 , -0.1 , -0.1 , -0.1 ],
         [-0.02, -0.1 ,  1.  , -0.1 , -0.1 , -0.1 , -0.1 ],
         [-0.25, -0.1 , -0.3 ,  1.  , -0.1 , -0.1 , -0.1 ],
         [-0.25, -0.1 , -0.3 , -0.1 ,  1.  , -0.1 , -0.1 ],
         [-0.25, -0.1 , -0.3 , -0.1 , -0.1 ,  1.  , -0.1 ],
         [-0.25, -0.1 , -0.3 , -0.1 , -0.1 , -0.1 ,  1.  ]])
# REVENUES array created by:
# x = np.eye(7) * 1.10 - 0.10  # 1 eur for hit, -10 cent for miss
# x[0, 1:] = -.20  # predicting a bird where there is none: -20 cent
# x[1:, 0] = -.25  # predicting no bird where there is one: -25 cent
# x[[1, 3, 4, 5, 6], 2] = -.3  # mistaking any bird for a pigeon: -30 cent
# x[2, 0] = -.02  # missing a pigeon: -2 cent

def compute_revenue(preds, targets):
    # first step: ignore all nonbird frames immediately before or after a call
    # this will also ignore holes of 1 or 2 frames within a long call sequence
    expanded = scipy.ndimage.maximum_filter1d(targets, 3, axis=0)
    ignore = (expanded != targets) & (targets == 0)
    targets = targets[~ignore]
    preds = preds[~ignore]
    # second step: compute confusion matrix
    confusions = sklearn.metrics.confusion_matrix(
            targets, preds,
            labels=np.arange(len(REVENUES)))
    # third step: compute revenue from confusions
    return (REVENUES * confusions).sum()