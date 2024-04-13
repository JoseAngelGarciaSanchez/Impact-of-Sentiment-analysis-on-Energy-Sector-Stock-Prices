import sys

from preprocessing.preprocessing import PreprocessorPipeline


path = sys.argv[1]
output_path = "./../data_cleaned/" + path.split("/")[-1]
preprocessing = PreprocessorPipeline(path=path, output_path=output_path)
preprocessing.launch()