import argparse, sys, os

"""
Preprocessor iterates over input file lines and filters out labels which 
are not needed
"""
class Preprocessor:
    def __init__(self, file_in_path):
        self.file_in_path = file_in_path

        # Set file_out_path - add appendinx to input file path
        file_in_path, file_in_extension = os.path.splitext(self.file_in_path)
        self.file_out_path = file_in_path + '-preprocessed' + file_in_extension

    def remove_rows(self, remove):
        # Open input and output files
        file_in = open(self.file_in_path, 'r')
        file_out = open(self.file_out_path, 'w')

        # Iterate over input file lines, if second column (label) not in 'remove' write it in the output file
        for sample in file_in.read().splitlines():
            if not sample.split(',')[1] in remove:
                file_out.write(sample + '\n')

        return self.file_out_path