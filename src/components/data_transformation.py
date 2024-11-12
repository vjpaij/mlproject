import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # -> for feature engineering
from sklearn.preprocessing import OneHotEncoder, StandardScaler # -> for feature engineering
from sklearn.impute import SimpleImputer # -> for imputing missing values
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

'''
In machine learning, a preprocessor.pkl file typically refers to a serialized object that contains preprocessing steps applied to the 
data before training a model. The .pkl extension stands for "pickle," a Python module that is used to serialize and save objects to 
a file, allowing them to be loaded and reused later.

Common Use Cases for preprocessor.pkl:
Feature Scaling: If you've applied transformations like standardization (using StandardScaler) or normalization (using MinMaxScaler), 
the preprocessor.pkl file might store the scaler fitted on the training data so it can be used to transform future data in the same 
way.

Encoding Categorical Variables: If you've used techniques like one-hot encoding (OneHotEncoder) or label encoding (LabelEncoder) to 
handle categorical features, the preprocessor.pkl might store the encoder objects, ensuring that categorical features are 
consistently encoded for both training and inference.

Imputation of Missing Values: If you've imputed missing values using a method like mean imputation, the preprocessor.pkl file could 
contain the imputation strategy or the fitted imputer, allowing you to apply the same imputation to new data.

Pipeline: In more complex workflows, a preprocessor.pkl file might store a complete preprocessing pipeline (using Pipeline from 
sklearn), which contains multiple steps (e.g., scaling, encoding, and feature selection) that should be applied sequentially to the 
data.

Why Save a Preprocessor?
Consistency: By saving the preprocessor, you ensure that the same transformations are applied to new data in the future, whether it's 
validation data or real-world data when you deploy the model.
Avoid Re-fitting: Instead of re-calculating scaling parameters, encodings, or imputation strategies every time you process new data, 
you can load the preprocessor from the saved file and apply it directly to new data.

Conclusion
preprocessor.pkl is a file that stores preprocessing steps, making it easier to maintain consistency and avoid redundant 
calculations, especially when applying transformations to new data or during model deployment.
'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            logging.info('Numerical Pipeline to correct missing data and to Standard Scale started')
            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Categorical Pipeline to correct missing data, Encoding and to Standard Scale started')
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',numerical_pipeline,numerical_columns),
                    ('cat_pipeline',categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info('Reading Train and Test datasets')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"

            '''
            axis=1: Refers to columns. When you use drop with axis=1, it removes the specified column(s) from the DataFrame.
            axis=0 (default): Refers to rows. If axis=0 were specified, it would drop rows instead of columns.
            '''
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying prepocessing on training and test dataframes')
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            '''
            np.c_: This is a shorthand for column-wise concatenation in NumPy. It stacks arrays as columns to create a larger 
            2D array, which is especially useful when you have features in one array and target labels in another and want to 
            combine them into a single array.
            '''
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            '''
            In Python, a pickle file is used for serializing and deserializing objects. This process is known as “pickling” and 
            “unpickling.” Here’s a brief overview:
            Pickling: This converts a Python object hierarchy into a byte stream. This byte stream can then be written to a file, 
            allowing you to save the state of an object.
            Unpickling: This converts the byte stream back into the original Python object hierarchy, effectively restoring the 
            object.
            Pickle is particularly useful for saving complex data structures like lists, dictionaries, or custom objects to a file, 
            and then loading them back into your program later
            For complex objects we use dill instead of pickle.
            The below save_object function in your code snippet is used to serialize and save an object to a file
            '''
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


