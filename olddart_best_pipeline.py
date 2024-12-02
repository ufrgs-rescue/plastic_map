import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
selected_columns = ['Label', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR1', 'NIR2', 'SWIR1', 'SWIR2', 'NDWI', 'WRI', 'NDVI', 'AWEI', 'MNDWI', 'SR', 'PI', 'RNDVI', 'FDI', 'PWDI']
tpot_data = pd.read_csv('files/csv_files/dataset_dart_2021_bilinear.csv', sep=',', dtype=np.float64, usecols=selected_columns)
features = tpot_data.drop(['Label', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Label'], random_state=42)

# Average CV score on the training set was: 1.0
exported_pipeline = XGBClassifier(learning_rate=0.001, max_depth=9, min_child_weight=7, n_estimators=100, n_jobs=1, subsample=0.45, verbosity=0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

#exported_pipeline.fit(training_features, training_target)
#results = exported_pipeline.predict(testing_features)
#print(results)