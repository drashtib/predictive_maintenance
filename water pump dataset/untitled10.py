# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 01:27:55 2021

@author: Drashti Bhatt
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

# Files to read
csv_data_sets = ['Water_Training_set_values.csv', 'Water_Training_set_labels.csv', 'Water_Test_set_values.csv']
# used to store dataframes: train_features_df, train_label_df, test_features_df
df_list = [] 

df_i = 0
# Read in the training values, training labels and testing data sets
for i in csv_data_sets:
    try:
        data_file = 'raw-data\\'+ i
        df_list.append(pd.read_csv(data_file))
        print("Water pump dataset {} has {} samples with {} features each".format(data_file, *df_list[df_i].shape))
    except:
        print("Water pump dataset {} could not be loaded. Is the dataset missing?".format(data_file))
    df_i+=1

# set up the data frames that we will be using to perform EDA 
train_features_df = df_list[0]
train_label_df = df_list[1]
test_features_df = df_list[2]
print("Confirm shapes of the train and test dataframes:",train_features_df.shape, train_label_df.shape, test_features_df.shape)

#%% Explore the features of the train set and  dtypes - 40 features 
#check the categorical features
categorical_features_df = train_features_df.select_dtypes(include=['object']).copy()

# Store the list of categorical features
categorical_cols = categorical_features_df.columns.tolist()
# Remove dates features
categorical_cols = categorical_cols[1:] 
#  look at the cardinality of categorical features
categorical_features_df.nunique().sort_values(ascending=False)

#sample the categorical features
for i in categorical_cols:
    print ("***** \'{}\' feature has \'{}\' category type *****".format(i,categorical_features_df[i].nunique()))
    print(categorical_features_df[i].unique())
    
# Categorical features of interest
sample_categorical_features = ['waterpoint_type', 'water_quality', 'source_type','region', 'management_group']
for i in sample_categorical_features:
    print("***** \'{}\' feature distribution breakdown *****".format(i))
    #print(categorical_features_df[i].value_counts())
    sns.set(style='darkgrid')
    fig = plt.figure(figsize = (20,4))
    sns.countplot(x=i,data=categorical_features_df, order = categorical_features_df[i].value_counts().index)
    plt.show()

#check the categorical features
numerical_features_df = train_features_df.select_dtypes(include=['int64','float64']).copy()

# Store the list of numerical features
numerical_cols = numerical_features_df.columns.tolist()
# total static head stats profile
numerical_features_df['amount_tsh'].describe()

plt.figure(figsize=(12, 7))
sns.boxplot(x=numerical_features_df['amount_tsh'])
#train_label_df['status_group'].value_counts()
sns.countplot(x='status_group',data=train_label_df)

#missing training data
plt.figure(figsize=(20, 7))
sns.heatmap(train_features_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.figure(figsize=(20, 7))
sns.heatmap(train_features_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(20, 7))
sns.heatmap(train_features_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# now check for duplicates
train_features_df.duplicated().sum()

#train_label_df
num_null_val = train_label_df.isnull().sum().sort_values(ascending=False).head(20)
num_null_val
#train_features_df.duplicated().sum()

# get an overview of the missing test data fields (NaN) in yellow
plt.figure(figsize=(20, 7))
sns.heatmap(test_features_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#test_features_df
num_null_val = test_features_df.isnull().sum().sort_values(ascending=False).head(20)
num_null_val

# Save panda dfs in raw-data
train_features_df.to_pickle('raw-data\pump_train_features_df.pkl')
train_label_df.to_pickle('raw-data\pump_train_label_df.pkl')
test_features_df.to_pickle('raw-data\pump_test_features_df.pkl')

#%% Modelling
import mxnet as mx
import keras
from keras import metrics
from keras import utils
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier    

# Show the deep learning library versions
print ("MXNet version = {}".format(mx.__version__))
print ("Keras version = {}".format(keras.__version__))

class ModelManager:
    def __init__(self,
                 model_dict={},
                 benchmark_model="",
                 label_names="",
                 num_procs=-1):
        '''creates model dictionary and manager object'''
        print("ModelManager: __init__ ...")

        self.model_dict = model_dict
        self.model_results = {}
        self.benchmark_model = benchmark_model
        self.best_model = None
        self.predictions = None
        self.scoring = f1_score # accuracy_score  # f1_score # 'f1', average=’macro’ #roc_auc # recall_score
        self.test_train_ratio = 0.2  # train and validation set -> 70% 30% 
        self.num_procs = num_procs
        self.label_names = label_names
        print("ModelManager: __init__ Number of models {}".format(len(self.model_dict)))
        self._print_models()

    def _add_model(self, new_model_pair):
        '''add a model to the  manager object  '''
        print("ModelManager: _add_model ...")
        self.model_dict.update(new_model_pair)
        self._print_models()
               
    def _print_models(self):
        '''print the models stored in the model manager object  '''
        for i in self.model_dict:
            print ("****** {} ****** \n {} \n ".format(i,self.model_dict[i]))
    
    def _print_model_scores(self):
        '''print the scores of the modesl eveluated '''
           print("ModelManager: _print_model_scores ...")
        for i in wp_models.model_results:
            print("{} : {:.3f}".format(i, self.model_results[i]))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(self.model_results.keys(), self.model_results.values(), width = 0.7)
        plt.show()

    def _make_scorer(self):
        '''build a custom scorer to evalute models '''
        print("ModelManager: _make_scorer ... {}".format(self.scoring))
        return (make_scorer(self.scoring, average='weighted'))

    def _cross_validate(self, wp_data_instance, k_fold=10):
        '''fold cross validate models on the training dataframe in Water_Asset_Data object '''
        print("ModelManager: _cross_validate ...")

        target_df = wp_data_instance.train_feature_df[wp_data_instance.label_col]
        feature_df = wp_data_instance.train_feature_df.drop(
            [wp_data_instance.id_col, wp_data_instance.label_col], axis=1)
        custom_scorer = self._make_scorer()
        print("ModelManager: _cross_validate feature df shape = {} and target df shape = {}".
            format(feature_df.shape, target_df.shape))

        for model_name, model in self.model_dict.items():
            cv_results = cross_val_score(
                model,feature_df,target_df,cv=k_fold,
                scoring=custom_scorer,n_jobs=self.num_procs)
            self.model_results[model_name] = cv_results.mean()
            print("ModelManager: _cross_validate => {} score for model {}: {:.3f} +/- {:.3f}".format(
                self.scoring, model_name, cv_results.mean(),cv_results.std()))
            
    def _fit_model(self, model_name, wp_data_instance):
        ''' fit _fit_model model on the train dataframe (train_test_split) in Water_Asset_Data object '''
        print("ModelManager: _fit_model ...")

        target_df = wp_data_instance.train_feature_df[wp_data_instance.label_col]
        feature_df = wp_data_instance.train_feature_df.drop(
            [wp_data_instance.id_col, wp_data_instance.label_col], axis=1)
        model = self.model_dict[model_name]

        # Split the data into a train and a test set
        # stratify param makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df,
            target_df,
            test_size=self.test_train_ratio,
            stratify=target_df,
            shuffle=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        np.set_printoptions(precision=2)
        # normalized confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fit_score = self.scoring(y_test, predictions, average='weighted')
        if model_name not in self.model_results:
            self.model_results[model_name] = fit_score
        
        print("Overall {} Score for model {} = {:.3f}".format(
            self.scoring, model_name,fit_score))
        print("Overall Accuracy Score for model {} = {:.3f}".format(
            model_name, accuracy_score(y_test, predictions)))
        print("Normalized confusion matrix for model {}".format(model_name))
        print(cm)
        print("Classification report for model {}".format(model_name))
        print(classification_report(y_test, predictions))
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(model.feature_importances_,list(X_train)).sort_values(ascending=False).head(20)
            feature_importance.plot(kind='bar', title=(model_name + ': Importance of Features'))

    def _select_best_model(self):
        '''select model with highest score'''
        print("ModelManager: _select_best_model ...")

        best_model_name = max(self.model_results, key=lambda key: self.model_results[key])
        self.best_model = self.model_dict[best_model_name]
        print("ModelManager: _select_best_model - best model is {}".format(best_model_name))
        self._print_model_scores()
        return (best_model_name)

    def _predict(self, wp_data_instance):
        '''generate predictions with best model and data frame parameters'''
        print("ModelManager: _predict ... with model {}".format(self.best_model))
        test_feature_df = wp_data_instance.test_feature_df.drop(wp_data_instance.id_col, axis=1)
        self.predictions = self.best_model.predict(test_feature_df)

        predictions_series = pd.Categorical(pd.Series(self.predictions))
        wp_data_instance.test_feature_df[wp_data_instance.label_col] = predictions_series
        return(self.predictions)
        
    def _get_predictions_df(self, wp_data_instance):
        '''return prediction data frame with index'''
        return(wp_data_instance.test_feature_df[[wp_data_instance.id_col, wp_data_instance.label_col]])

#%%Load water pump data object
!dir clean-data\*.pkl
filename = 'clean-data/clean_wp_data_object.pkl'
infile = open(filename,'rb')
wp_data = pickle.load(infile)
wp_data.train_feature_df.shape

#%%Create the different tuned models to compare
#We will use a logistic regression as a benchmark model
#we will then use a random forest and, boosted and XG boost models to evaluate results against the benchmark

num_procs = -1
k_fold = 5

# Benchmark Model define a pipeline with PCA and logistic regression as our baseline Logistic regression: ovr = one (class) 
# versus rest (of classes)
lr_std_pca = make_pipeline(
    StandardScaler(), PCA(),
    LogisticRegression(multi_class='ovr', class_weight='balanced', n_jobs=num_procs))

# Random Forest ensemble model - previously manually tuned
random_forest = RandomForestClassifier(
    n_estimators=250,
    max_features='sqrt',
    min_samples_split=2,
    max_depth=20,
    min_samples_leaf=1,
    class_weight = 'balanced',
    n_jobs=num_procs)

# GBM ensemble model - previously manually tuned
gradient_boosting = GradientBoostingClassifier(
    criterion='friedman_mse',
    learning_rate=0.1, loss='deviance', max_depth=12,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=200,
    presort='auto', verbose=0)


xg_boosing = XGBClassifier(
 eta = 0.2, booster = "gbtree",
 n_estimators=500,  max_depth=12,
 min_child_weight=1,  gamma=0,
 num_class = 3, objective= 'multi:softmax',
 n_jobs=num_procs,  scale_pos_weight=1)

benchmark_model = 'Logistic Regresstion Std PCA' # 'lr_std_pca'
model_dict = { benchmark_model: lr_std_pca, 
              'Random Forest': random_forest, 
              'Gradient Boosting': gradient_boosting,
              'XG Boost': xg_boosing}

#%%Create the model manager object which will store the models
wp_models = ModelManager(model_dict=model_dict, benchmark_model=benchmark_model,
                         label_names=wp_data.label_names, num_procs=num_procs)

#%%Evaluate models and fit the best performing model

wp_models._cross_validate(wp_data, k_fold=k_fold)
best_model_name = wp_models._select_best_model()
wp_models._fit_model(best_model_name, wp_data)

#%%Evaluate a Neural Network and compare to the current best model
# create a neural net model - 3 layers, previously tuned
from keras import regularizers
def create_nn_model(input_shape, output_shape):
    print("create_model() with input dim = {} and output dim = {}".format(
        input_shape, output_shape))

    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, input_dim=input_shape, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))
    # Compile model, use GPU
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'], context=['gpu(0)'])

    print(model.summary())
    return model

# used the neuralnet hyper parameters previsouly used
input_dim = wp_data.train_feature_df.drop([wp_data.id_col, wp_data.label_col], axis=1).shape[1]
output_dim = wp_data.train_feature_df[wp_data.label_col].nunique()
n_epoch = 100
batch_sz = 16
verbose = 1

# Build a neural net pipeline
nn_model = KerasClassifier(build_fn=create_nn_model, input_shape=input_dim,
                           output_shape=output_dim,  epochs=n_epoch, batch_size=batch_sz, verbose=verbose)

nn_model_pipeline = make_pipeline(StandardScaler(), nn_model)

nn_model_dict = {'nn_model_pipeline': nn_model_pipeline}

# add the neural network model to the ModelManager Object
wp_models._add_model({'nn_model_pipeline':nn_model_pipeline})
wp_models._fit_model('nn_model_pipeline', wp_data)
best_model_name = wp_models._select_best_model()

#%%Generate and submit predictions performed by the best performing model on the test data set

predicitions = wp_models._predict(wp_data)
len(predicitions)
# def _get_predictions_df(self, label_df, label_id, label_col):
prediction_df = wp_models._get_predictions_df(wp_data_instance = wp_data)
prediction_df.head()

# feature names to be mapped
gps_map_features = ['latitude', 'longitude']

# create a map object
wp_map = gis_map.GIS_Map_Viz(latitude_feature_name = gps_map_features[0], longitude_feature_name = gps_map_features[1],
                             gps_bounderies_dict = wp_data.gps_bounderies_dict)

#Display the test set color coded water points
wp_map._display_gps_map(wp_data.test_feature_df, prediction_df[wp_data.label_col],
                         "Test set predictions - Water Points in Tanzania")

sumbission_data_dir = 'prediction-data\\'
submission_file = sumbission_data_dir  + 'pump_predictions_submission_df.csv'

prediction_df.to_csv(submission_file, index=False)

from IPython.display import Image
Image(filename='Competition GBM Results.png')