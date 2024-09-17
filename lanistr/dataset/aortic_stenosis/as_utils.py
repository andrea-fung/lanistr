import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio as iio
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from typing import List, Dict, Union, Tuple


def fix_leakage(df, df_subset, split='train'):
    """Fix overlap in studies between the particular 'split' subset and the other subsets 
    in the dataset.
        
        Args:
        df: DataFrame of the complete dataset. 
        df_subset: A view of a subset of the DataFrame; it is either the
            train, validation or test set.
        split: Whether the df_subset is associated with the train/val/test set

        Returns:
        A dataframe of df without any data leakage problem between the train, val, test 
        subsets.
    """
    train = df[df['split']=='train']
    val = df[df['split']=='val']
    test = df[df['split']=='test']

    #Check whether any study ID in one subset is in any other subset
    val_test = val['Echo ID#'].isin(test['Echo ID#']).any()
    train_test = train['Echo ID#'].isin(test['Echo ID#']).any()
    train_val = train['Echo ID#'].isin(val['Echo ID#']).any()
    print("Checking if there is data leakage...")
    print(f"There is overlap between: val/test: {val_test}, train/test: {train_test}, train/val: {train_val}")

    #Get indices for all rows in the subset that overlap with another subset
    train_test_overlap = train['Echo ID#'].isin(test['Echo ID#'])
    train_test_leak_idx = [i for i, x in enumerate(train_test_overlap) if x]

    val_test_overlap = val['Echo ID#'].isin(test['Echo ID#'])
    val_test_leak_idx = [i for i, x in enumerate(val_test_overlap) if x]
    
    #Get unique study IDs corresponding to the overlapping rows
    train_test_leak_ids = train['Echo ID#'].iloc[train_test_leak_idx].to_list()
    train_test_leak_ids = list(set(train_test_leak_ids))

    val_test_leak_ids = val['Echo ID#'].iloc[val_test_leak_idx].to_list()
    val_test_leak_ids = list(set(val_test_leak_ids))

    print(f"Echo IDs of overlapping studies between: val/test: {val_test_leak_ids}, train/test: {train_test_leak_ids}")

    #Assign overlapping studies to only one subset
    num_remove_test = len(train_test_leak_ids)//2
    remove_test_ids = train_test_leak_ids[0:num_remove_test]
    remove_train_ids = train_test_leak_ids[num_remove_test:]  

    num_remove_val = len(val_test_leak_ids)//2
    remove_val_ids = val_test_leak_ids[0:num_remove_val]
    remove_test_ids = remove_test_ids + val_test_leak_ids[num_remove_val:]  

    if split == 'train':
        fixed_subset = remove_ids(remove_ids=remove_train_ids, dataset=df_subset)
        if len(fixed_subset) == len(df_subset) - 5:
            print("Data leakage for train/test subsets has been fixed.")
    elif split == 'val':
        fixed_subset = remove_ids(remove_ids=remove_val_ids, dataset=df_subset)
    elif split == 'test':  
        fixed_subset = remove_ids(remove_ids=remove_test_ids, dataset=df_subset)
        if len(fixed_subset) == len(df_subset) - 8:
            print("Data leakage for train/test subsets has been fixed.")
    
    return fixed_subset

def remove_ids(remove_ids, dataset):
    "Remove rows with 'Echo ID#' in the list of remove_ids for the dataset"
    for id in remove_ids:
        remove_rows = dataset[dataset['Echo ID#']==id].index.values
        dataset = dataset.drop(index=remove_rows)
    
    return dataset

#utils from tabular transformer finetuning branch
def preprocess_as_data(train, val, test, cat_cols):

    #train = train.replace(-1, np.nan)
    #val = val.replace(-1, np.nan)
    #test = test.replace(-1, np.nan)
    
    # Remove non numerical features
    numeric_feats = train.columns.to_list().copy()
    numeric_feats = [col for col in numeric_feats if col not in cat_cols]
    numeric_feats.remove('as_label')

    # Replace any missing values in the categorical columns with "VV_likely"
    # train[cat_cols] = train[cat_cols].fillna(-1)
    # val[cat_cols] = val[cat_cols].fillna(-1)
    # test[cat_cols] = test[cat_cols].fillna(-1) 

    processor = make_column_transformer(
        (GaussianImputerGivenMean(strategy='mean', mean=1.5, std=0.3), ['VPeak']),
        remainder='passthrough'
    )

    # Create a new list without 'VPeak' - this is a current hack for the imputation and can be cleaned up
    new_numeric_feats = [feat for feat in numeric_feats if feat != 'VPeak']
    all_columns = ['VPeak'] + new_numeric_feats + cat_cols

    pipe = make_pipeline(StandardScaler(), GaussianImputer())
    processor2 = make_column_transformer(
        (pipe, numeric_feats),
        remainder='passthrough'
    )

    # get normalized versions and PeakV of each set for numeric features only
    train_new = train[all_columns]
    val_new = val[all_columns]
    test_new = test[all_columns]

    train_temp = pd.DataFrame(processor.fit_transform(train_new), columns=all_columns, index=train.index)
    val_temp = pd.DataFrame(processor.transform(val_new), columns=all_columns, index=val.index)
    test_temp = pd.DataFrame(processor.transform(test_new), columns=all_columns, index=test.index)

    # get imputed versions of each set for numeric features only
    # imputation for Peak gradient
    train_temp = fill_peak_gradient(train_temp)
    val_temp = fill_peak_gradient(val_temp)
    test_temp = fill_peak_gradient(test_temp)

    #iterative imputation for the remaining columns?
    train_impute = pd.DataFrame(processor2.fit_transform(train_temp), columns=all_columns, index=train.index)
    val_impute = pd.DataFrame(processor2.transform(val_temp), columns=all_columns, index=val.index)
    test_impute = pd.DataFrame(processor2.transform(test_temp), columns=all_columns, index=test.index)

    #revert column order back to original 
    og_col_order = train.columns.to_list()[1:] #remove first column --> as_label 
    train_impute = train_impute[og_col_order]
    val_impute = val_impute[og_col_order]
    test_impute = test_impute[og_col_order]

    # train = train.replace(np.nan, -1)
    # val = val.replace(np.nan, -1)
    # test = test.replace(np.nan, -1)

    # create a dataset with each of these
    # train_set = ASDataset(train, train_impute, all_columns)
    # val_set = ASDataset(val, val_impute, all_columns)
    # test_set = ASDataset(test, test_impute, all_columns)

    return (train_impute, val_impute, test_impute, all_columns)

def load_as_data(csv_path: str,
                    drop_cols : Union[List[str], None] = None,
                    num_ex : Union[int, None] = None,
                    test_split : float = 0.1,
                    random_seed : Union[int, None] = None,
                    scale_feats : bool = True
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        """Processes data for imputation models.

        Imputes missing values with average of feature and optionally drops columns and scales all 
        data to be normally distributed.
        
        Args:
            csv_path: Path to the dataset to use. Should be a csv file.
            drop_cols: List of columns to drop from dataset. If None, no columns are dropped.
            num_ex: Number of examples to use. If None, will use all available examples.
            test_split: What fraction of total data to use in test set. Also used to split
                validation data after test data has been separated.
            random_seed: Seed to initialize randomized operations.
            scale_feats: Whether to scale numeric features in dataset during preprocessing.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset).
            Each is a processed pandas DataFrame.

        Raises:
            Exception: Specified more examples to use than exist in the dataset.
        """

        data_df = pd.read_csv(csv_path, index_col=0)
        data_df = data_df.drop("AV stenosis", axis=1)
        #data_df = data_df.drop("age", axis=1)

        #If num_ex is None use all examples in dataset
        if not num_ex:
            num_ex = data_df.shape[0]

        #Ensure number of examples specified is not greater than examples in the dataset
        elif num_ex > data_df.shape[0]:
            ex_string = "Specified " + str(num_ex) + " examples to use but there are only " + str(nan_df.shape[0]) + " examples with known target column in dataset."
            raise Exception(ex_string)
        

        #Create description of processing and store
        print("Processing data from: " + csv_path + "\n")
        print("Dropping the following columns: " + str(drop_cols) + "\n")
        print("Using " + str(num_ex) + " examples with test split of " + str(test_split) + ".\n")
        print("Random seed is " + str(random_seed) + ".\n")
        print("Scaling features? " + str(scale_feats) + "\n")

        #Replace any -1 values with NaNs for imputing
        #nan_df = data_df.replace(-1, np.nan)

        #Sample data to only contain num_ex rows
        sampled_df = data_df.sample(n=num_ex, random_state=random_seed)

        #If drop columns is not empty, drop specified. Otherwise keep DataFrame as is
        drop_cols_df = sampled_df.drop(columns=drop_cols) if drop_cols else sampled_df

        #Get tabular information 
        drop_cols_df, input_dim, cat_idxs, cat_dims = encode_tab_features(drop_cols_df)

        #Split into train, test, and validation sets
        train_df = drop_cols_df[drop_cols_df['split'] == 'train'].drop(columns=['split'])
        val_df = drop_cols_df[drop_cols_df['split'] == 'val'].drop(columns=['split'])
        test_df = drop_cols_df[drop_cols_df['split'] == 'test'].drop(columns=['split'])

        print("\nTrain dataset shape:", train_df.shape)
        print("Validation dataset shape:", val_df.shape)
        print("Test dataset shape:", test_df.shape,"\n")

        return ((train_df, val_df, test_df), (input_dim, cat_idxs, cat_dims))

def encode_tab_features(
    data: pd.DataFrame, #categorical_cols: List[str], numerical_cols: List[str]
) -> Tuple[pd.DataFrame, List[int], List[int], int]:
  """Encodes tabular features for machine learning processing.

  This function handles both categorical and numerical features in a DataFrame:

  - Categorical features are label-encoded, with missing values filled as
  "VV_likely".

  Args:
      data: The pandas DataFrame containing the features.
      categorical_cols: A list of column names representing categorical
        features.
      numerical_cols: A list of column names representing numerical features.

  Returns:
      A tuple containing:
          - The modified DataFrame with encoded features.
          - A list of indices indicating the positions of categorical features.
          - A list of dimensions (number of unique values) for each categorical
          feature.
          - The total input dimension (number of features after encoding).
  """
  categorical_topics = ['AV ', 'MV', 'RV', 'Bicuspid', 'Sclerotic']
  categorical_columns = []
  categorical_dims = {}
  for col in data.columns:
    if data[col].dtype=='int64' or any(t in col for t in categorical_topics): 
        data.loc[:, col] = data.loc[:, col].fillna(-1)
        l_enc = preprocessing.LabelEncoder()
        data.loc[:, col] = l_enc.fit_transform(data[col].values) #fit transform includes missing val as new cat
        categorical_columns.append(col)
        unique_classes = data[col].unique()
        
        categorical_dims[col] = len(l_enc.classes_) #includes missing val as a class

  input_dim = len(data.columns) - 2 # minus split and AS_label
  features = data.columns[2:] # minus split and AS_label
  cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
  cat_dims = [
      categorical_dims[f]
      for _, f in enumerate(features)
      if f in categorical_columns
  ]

  return data, input_dim, cat_idxs, cat_dims

def fill_peak_gradient(df):
    # Find rows where 'AoPG' is NaN
    missing_ao_rows = df['AoPG'].isna()

    # Replace NaN values in 'AoPG' with the calculated values based on the formula
    df.loc[missing_ao_rows, 'AoPG'] = 4 * (df.loc[missing_ao_rows, 'VPeak']**2)

    return df

class GaussianImputer(TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        self.means_ = np.nanmean(X, axis=0)
        self.stddevs_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        X = check_array(X, force_all_finite=False)

        for i in range(X.shape[1]):
            nan_mask = np.isnan(X[:, i])
            num_missing = np.sum(nan_mask)
            if num_missing > 0:
                random_values = np.random.normal(loc=self.means_[i], scale=self.stddevs_[i], size=num_missing)
                X[nan_mask, i] = random_values

        return X
    
class GaussianImputerGivenMean(SimpleImputer):
    def __init__(self, strategy='mean', fill_value=None, mean=None, std=None, **kwargs):
        self.mean = mean
        self.std = std
        super().__init__(strategy=strategy, fill_value=fill_value, **kwargs)

    def transform(self, X):

        # Get the indices of missing values in the column
        missing_indices = np.where(np.isnan(X))[0]

        # Generate random samples from a Gaussian distribution around mean=1.5
        fill_values = np.random.normal(loc=self.mean, scale=self.std, size=len(missing_indices))

        # Clip values to ensure they are not less than 0
        fill_values = np.clip(fill_values, 0, None)

        # Reshape the fill_values array to be a column vector
        fill_values = fill_values.reshape(-1, 1)

        # Create a new DataFrame and replace the original one
        new_X = X.copy()
        new_X.iloc[missing_indices] = fill_values

        return new_X


