import argparse

from preprocessor import Preprocessor
from feature_collector import FeatureCollector
from classifier import Classifier

class MachineLearningPipeline:
    def __init__(self, file_in_path, remove):
        self.file_in_path = file_in_path
        self.remove = remove

    def run(self):
        # 1. Step of ML Pipeline: Preprocess input file
        print ':: Preprocessing \t\t%s' % (self.file_in_path)
        preprocessor = Preprocessor(self.file_in_path)
        file_preprocessed_path = preprocessor.remove_rows(self.remove)

        # 2. Step of ML Pipeline: Collect features
        print ':: Collecting features in \t%s' % (file_preprocessed_path)
        feature_collector = FeatureCollector(file_preprocessed_path)
        file_features_path = feature_collector.collect_features()

        # 3. Step of ML Pipeline: Classification
        print ':: Classifying samples in \t%s' % (file_features_path)
        classifier = Classifier(file_features_path)
        classifier.classify()
        print ':: Saving model'
        # classifier.save()

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help='Path to the input file')
parser.add_argument('--remove', '-r', action='append', help='What labels should be removed')
args = parser.parse_args()

# Initialize and run MachineLearningPipeline
learning_pipeline = MachineLearningPipeline(args.file, args.remove)
learning_pipeline.run()