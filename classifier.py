import argparse, os, sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals import joblib

"""
Classifier takes features dataset as an input and splits it into training 
and testing data, trains Random Forest classifier, and predicts labels on 
the training set. It evaluates the results - return classification report, 
confusion matrix, feature importances - and visualize one decision tree from 
the random forest.
"""
class Classifier:
    def __init__(self, file_features_path):
        self.file_features_path = file_features_path

    def classify(self):
        # Open input file
        file_in = (self.file_features_path)

        # Read features
        features_names = ('swear_word_contained', 'offensiveness_score', 
            'urls_count', 'user_mentions_count', 'letters_count', 
            'upper_letters_sum', 'ratio_letters_upperletters', 'polarity', 
            'subjectivity', 'retweets_count', 'favourite_count', 
            'friends_count', 'followers_count', 'ratio_followers_friends', 
            'total_posts_count', 'account_age')
        labels_names = ['abusive', 'hateful', 'normal']
        # Drop unnecessary features (columns 2, 4, 6 and 19) from the dataset
        features = pd.read_csv(file_in, names=features_names, 
            usecols = [1,3,5,20] + range(7,19))
        labels = pd.read_csv(file_in, header = None, usecols = [21])
        
        features = features.values
        labels = labels.values
        
        # Inicialize KFold class
        kf = KFold(n_splits=10)

        # Inicialize RFC class
        clf = RandomForestClassifier()
        
        precision_classes = []
        precision_total = [] 
        recall_classes = []
        recall_total = []
        fscore_classes = []
        fscore_total = []
        importances_sum = []
        confusion_sum = []
        # Iterate n_split times over features field
        for train_index, test_index in kf.split(features):  

            # Divide features and labels into train data and test data 
            features_train, features_test = features[train_index], 
                features[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # Fit data in Random Forest
            clf.fit(features_train, labels_train)
            # Predict
            prediction = clf.predict(features_test)
            
            # Classification report for each split
            precision_classes.append(precision_score(labels_test, prediction,
                average=None))
            precision_total.append(precision_score(labels_test, prediction,
                average='weighted'))
            recall_classes.append(recall_score(labels_test, prediction,
                average=None))
            recall_total.append(recall_score(labels_test, prediction,
                average='weighted'))
            fscore_classes.append(f1_score(labels_test, prediction,
                average=None))
            fscore_total.append(f1_score(labels_test, prediction,
                average='weighted'))

            # Feature importances for each split
            importances_all = (clf.feature_importances_)
            importances_sum.append(importances_all)

            # Confusion Matrix for each split
            confusion_all = confusion_matrix(labels_test, prediction)
            confusion_sum.append(confusion_all)

        # Classification report average
        precision = np.mean(precision_classes, axis=0).tolist() + 
            [np.mean(precision_total,axis=0)]
        recall = np.mean(recall_classes, axis=0).tolist() +
            [np.mean(recall_total,axis=0)]
        fscore = np.mean(fscore_classes, axis=0).tolist() + 
            [np.mean(fscore_total,axis=0)]
        report_array = np.asarray([precision, recall, fscore])
        report = pd.DataFrame(report_array, ['precision', 'recall', 'f1-score'], 
            [labels_names + ['avg/total']])
        print report

        # Feature importances average
        importances_average = np.mean(importances_sum, axis=0).tolist()
        for feature, importance in zip(features_names, importances_average)
            feature_importances = (feature, round(importance, 4))
        feature_importances = sorted(feature_importances, key = lambda x: x[1],
            reverse=True)

        for pair in feature_importances:
            print 'Feature: {:30} Importance: {}'.format(*pair)

        # Confusion matrix average
        confusion_matrix_average = np.mean(confusion_sum, axis=0)
        print confusion_matrix_average

        # Visualize one tree from random forest
        one_tree = clf.estimators_[1]
        export_graphviz(one_tree, out_file='one_tree.dot', max_depth=5, 
            feature_names=features_names, class_names=labels_names, rounded = True, 
            precision=1)

        # Save classifier
        joblib.dump(clf, 'Classifier.pkl')