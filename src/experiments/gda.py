import sys
import numpy as np

sys.path.append('/Users/jonathan/cs229_final_project/FFML')

from src.util import dataprocessing as dp


rb_train_path = "all_rb_stats.csv"

x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(rb_train_path)