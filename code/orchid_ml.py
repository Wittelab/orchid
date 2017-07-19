# For progress display in ipython notebook
from IPython.core.display import clear_output
from IPython.display import clear_output

import sys, os, datetime, time
from random import shuffle
from collections import defaultdict

# Data visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color
from matplotlib.pylab import cm


# Scikit learn stuff
from sklearn import svm, linear_model                                                                    # SVM and linear models
from copy import deepcopy                                                                                # To duplicate models
from sklearn.externals import joblib                                                                     # To save jobs
from sklearn.ensemble import RandomForestClassifier                                                      # Random Forest model
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, ShuffleSplit        # Cross validation, parameter tuning
from sklearn.metrics import *                                                                            # To assess model performance
from sklearn.multiclass import OneVsRestClassifier                                                       # For multi class problems
from sklearn.neighbors import LSHForest
#from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, LabelEncoder, label_binarize    # To process and normalize data
from sklearn.preprocessing import *
# Other category encoders
import category_encoders as ce

from scipy import interp
from urlparse import urlparse
import json
import dill


class MutationMatrix(pd.DataFrame):
    "Some cancer mutation and machine learning extensions for the pandas DataFrame"

    _metadata          = [
                          'chunk_size', 'db_uri', 'db_metadata', 'mutation_table', 'mutation_table_id', 'mutation_id_column', 'donor_id_column', 'annotation_columns', 'feature_categories', 'features', 
                          'label_column', 'quiet', 'loaded', 'encoded', 'collapsed', 'normalized', 'imputer', 'scaler','normalize_options', 'model', 'X_train', 'X_test', 'Y_train', 'Y_test'
                         ]
    
    chunk_size         = 10000
    db_uri             = None
    db_metadata        = None
    mutation_table     = None
    mutation_table_id  = None  # 
    mutation_id_column = 'mutation_id'
    donor_id_column    = 'donor_id'
    label_column       = None
    annotation_columns = None
    feature_categories = None
    features           = None
    feature_transforms = {}
    quiet              = True
    loaded             = False
    encoded            = False
    collapsed          = False
    normalized         = False    # Whether this MutationMatrix has been normalized
    imputer            = None     # The imputer used to normalize this MutationMatrix
    scaler             = None     # The scaler used to normalize this MutationMatrix
    model              = None
    X_train            = None
    X_test             = None
    Y_train            = None
    Y_test             = None

    normalize_options  = {
                           'nan_strat'      : 'median',
                           'scaler_strat'   : 'standard',
                         }


    @property
    def _constructor(self):
        return MutationMatrix
    
    @property
    def _constructor_sliced(self):
        return pd.Series

    # Some data, and/or metadata
    # TODO: Add column populator
    def __init__(self, *args, **kwargs):
        self.db_uri   = kwargs.pop('db_uri', None)
        self.quiet    = kwargs.pop('quiet', True)
        super(MutationMatrix, self).__init__(*args, **kwargs)

    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    def __getitem__(self, key):
        "BUGFIX: Series generated from MutationMatrix retain old values, so force pd.Series to flush cache"
        self._clear_item_cache()
        result = super(MutationMatrix, self).__getitem__(key)
        result._clear_item_cache()
        result = super(MutationMatrix, self).__getitem__(key)
        return result

    def _get_db_column_type(self, table, column):
        db = urlparse(self.db_uri).path.strip('/')
        syntax = "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s' AND COLUMN_NAME = '%s'" % (db, table, column)
        type_ = pd.read_sql(syntax, self.db_uri)['DATA_TYPE'][0]
        if 'int' in type_:
            return 'int'
        else:
            return 'char'

    def set_backing_database(self, db_uri):
        self.db_uri = db_uri
        self.load_metadata()

    def load_metadata(self):
        ## db_uri
        if self.db_uri==None:
            print "Plese select a database using set_backing_database(database_uri). Metadata will then be automatically loaded."
            return
        self.db_metadata = {}
        syntax = "SELECT metavalue FROM metadata WHERE metakey='features'"
        features = pd.read_sql(syntax, self.db_uri)['metavalue'][0]
        features = json.loads(features)
        self.db_metadata['features'] = features
        self.features = pd.Series(features.keys())
        syntax = "SELECT metavalue FROM metadata WHERE metakey='params'"
        params = pd.read_sql(syntax, self.db_uri)['metavalue'][0]
        params = json.loads(params)
        self.db_metadata['params'] = params
        self.mutation_table = params['mutation_table']
        self.mutation_table_id = params['mutation_table']+"_id"
        return


    def set_features(self, feature_list=None):
        if type(feature_list)!=type(None):
            self.features = pd.Index(list(feature_list))
        else:
            self.features = pd.Index(set(self.columns) - set(self.annotation_columns) - set([self.label_column]))
        #else:
        #    features = []
        #    for tbl in self.features:
        #        subfeatures = self.db_metadata['features'][tbl]
        #        if type(subfeatures) != list:
        #            subfeatures = [subfeatures]
        #        for subfeature in subfeatures:
        #            features.append(subfeature['column'].lower())
        #    self.features = features

    def add_labels(self, mapping):
        mapping_columns = set(pd.DataFrame(mapping).columns)
        label_column = mapping_columns.difference(set(list(self.columns)+[self.index.name]))
        if len(label_column)==0:
            print "It appears the label column has already been added."
            return
        if len(label_column)>1:
            print "To add a label column to the MutationMatrix, you must provide a dataframe with two columns: 1) column label and values that match the MutationMatrix, and 2) A column of values to merge into the MutationMatrix."
            return
        anchor_column = list(mapping_columns.difference(label_column))[0]
        label_column  = list(label_column)[0]
        if anchor_column == self.index.name:
            self._data = pd.DataFrame(self.merge(mapping, how='left', left_index=True, right_on=anchor_column).drop(anchor_column,1))._data
        else:
            self._data = pd.DataFrame(self.merge(mapping, how='left', left_on=anchor_column, right_on=anchor_column))._data
        self.label_column = label_column

    def set_label_column(self, col_name):
        if col_name not in self.columns:
            print "You must specify a column label that exists in the MutationMatrix, or merge a label column into the MutationMatrix using add_labels()"
            return
        if col_name in self.features:
            self.features = self.features.drop(col_name)

        self.label_column = col_name


    def set_normalize_options(self, nan_strat='zero', scaler_strat='mms'):
        self.normalize_options = {
                                   'nan_strat'    :  nan_strat,
                                   'scaler_strat' :  scaler_strat,
                                 }


    # Takes a peek at the MutationMatrix, taking half of the requested rows from the top and bottom
    def peek(self, size=1000):
        X = self.as_matrix()
        subset = X if np.shape(X)[0] < size else X[range(0,int(size/2.))+range(-int(size/2.),-1),:]
        ax = sns.heatmap(pd.DataFrame(subset), yticklabels=False, xticklabels=False, cmap="Blues")


    def load(self, *args, **kwargs):
        self.load_mutations(*args, **kwargs)
        self.load_features(*args, **kwargs)


    def load_mutations(self, ids=None, by='donor', number_real='all', number_simulated=None, quiet=None):
        # Check input
        quiet = self.quiet if quiet==None else quiet
        ## db_uri
        if self.db_uri==None:
            print "Plese select a database using the set_backing_database(database_uri) method."
            return
        ## number_real
        if (number_real==None) or ((type(number_real)==int) and (number_real<1)):
            print "You must request at least 1 real mutation"
        elif (type(number_real)!=int) and (number_real!='all'):
            print "number_real must be either a positive integer or 'all'."
            return
        ## number_simulated
        if number_simulated==None:
            pass
        elif type(number_simulated) == int and number_simulated < 1:
            print "You must request a positive number of simulated mutations."
            return
        elif type(number_simulated) == str:
                if number_simulated not in ('max', 'average', 'equal', 'all'):
                    print "If specifying the number of simulated mutations as text, use 'max', 'average', 'equal', or 'all'."
                    return
        ## by
        if by not in ('donor', 'mutation'):
            print "Mutation information can only be loaded by donor or mutation ids at this time."
            return
        ## ids
        if not (ids is None or (type(ids) == list and all([type(a) == str for a in ids])) or type(ids) == str):
            print "'ids' must be either a single value (string), a list of donor or mutation ids (strings), or 'None' to load all mutations from the database."
            return

        if by=='donor':
            by = self.donor_id_column
        else:
            by = self.mutation_id_column

        # Load metadata
        if not quiet: print "Loading database metadata"
        self.load_metadata()

        # Get the real mutations
        real_mutations = None
        if number_real == 'all':
            syntax = "SELECT * FROM %s WHERE is_simulated=0" % self.mutation_table
            if ids:
                syntax += " AND %s IN ('%s')" % (by, "', '".join([str(a) for a in ids]))
                if not quiet: print "Getting [%s] real mutations for the requested %s ids..." % (str(number_real), by)
                sys.stdout.flush()
            else:
                if not quiet: print "Getting [all] real mutations..." 
            real_mutations = pd.read_sql(syntax, self.db_uri)
            number_real = real_mutations.shape[0]
        else:
            syntax = "SELECT %s FROM %s WHERE is_simulated=0" % (self.mutation_table_id, self.mutation_table)
            if ids:
                syntax += " AND %s IN ('%s')" % (by, "', '".join([str(a) for a in ids]))
            # Get the mutations
            if not quiet: print "Getting [%s] real mutations... (randomly sampled)" % str(number_real)
            real_ids = pd.read_sql(syntax, self.db_uri)
            # Sample the number of requested
            replace = False
            if len(real_ids) < number_real:
                if not quiet: print "Fewer possible real mutations than requested, sampling with replacement..."
                replace = True
            real_ids = real_ids.sample(number_real, replace=replace)
            syntax = "SELECT * FROM %s WHERE %s IN (%s)" % (self.mutation_table, self.mutation_table_id, ", ".join([str(a) for a in list(real_ids[self.mutation_table_id])]))
            real_mutations = pd.read_sql(syntax, self.db_uri)

        # Get simulated mutations if requested
        sim_mutations = None
        if number_simulated != None:
            if number_simulated == 'all':
                if not quiet: print "Getting [all] simulated mutations..."
                syntax = "SELECT * FROM %s WHERE is_simulated=1" % (self.mutation_table)
                sim_mutations = pd.read_sql(syntax, self.db_uri)
            else:
                if number_simulated == 'equal':
                    number_simulated = real_mutations.shape[0]
                    if not quiet: print "Getting [%d] simulated mutations... (equal to the total number of real mutations)" % number_simulated
                elif number_simulated == 'average':
                    number_simulated = int(real_mutations[self.donor_id_column].value_counts().mean())
                    if not quiet: print "Getting [%d] simulated mutations... (equal to the average number of real mutations grouped by donor)" % number_simulated
                elif number_simulated == 'max':
                    number_simulated = int(real_mutations[self.donor_id_column].value_counts().max())
                    if not quiet: print "Getting [%d] simulated mutations... (equal to the number of real mutations for the donor with the most)" % number_simulated
                else:
                    if not quiet: print "Getting %d simulated mutations..." % number_simulated
                sys.stdout.flush()
                # Get the mutations 
                sim_ids = pd.read_sql("SELECT %s FROM %s WHERE is_simulated=1" % (self.mutation_table_id, self.mutation_table), self.db_uri)
                # Sample the number requested
                replace = False
                if len(sim_ids) < number_real:
                    if not quiet: print "Fewer possible simulated mutations than requested, sampling with replacement..."
                    replace = True
                sim_ids = sim_ids.sample(number_simulated, replace=replace)
                syntax = "SELECT * FROM %s WHERE %s IN (%s)" % (self.mutation_table, self.mutation_table_id, ", ".join([str(a) for a in list(sim_ids[self.mutation_table_id])]))
                sim_mutations = pd.read_sql(syntax, self.db_uri)
        mutations = pd.concat([real_mutations, sim_mutations])
        mutations = mutations.set_index(self.mutation_table_id)
        self._data = mutations._data
        self.annotation_columns = mutations.columns
        if not quiet: print "Done"
        return


    def _load_feature_data(self, mutations, quiet=None, feature_categories=None):
        quiet = self.quiet if quiet==None else quiet
        if feature_categories==None:
            print "You must specify a list of feature categories (feature tables) to load feature data from if calling _load_feature_data manually."
            return
        #####
        # Join featurized data
        #####
        ids = list(mutations.index)
        for i, tbl in enumerate(feature_categories):
            
            if not quiet:
                try:
                    clear_output(wait=True)
                except:
                    pass
                print "%2d/%2d [%3.1f%% complete]  Loading feature '%s'   %s \r" % (i+1, len(feature_categories), 100*((i+1)/float(len(feature_categories))), tbl, " "*100)
                sys.stdout.flush()
            # To temporarily store chunked mutations 
            chunk_mutations = None

            for i in xrange(0, len(ids)+1, self.chunk_size):
                # The current chunks ids
                chunk_ids = [str(a) for a in ids[i:(i+self.chunk_size-1)]]
                # Convert the ids to a MySQL like search string, depending on the table key type
                syntax = "SELECT * FROM feature_%s WHERE %s IN (%s)" % (tbl, self.mutation_table_id, ", ".join(chunk_ids))
                temp   = pd.read_sql(syntax, self.db_uri)
                # Make sure the join ids are ints for the encoded_mutations merge (otherwise get nothing but NaNs!)
                temp[self.mutation_table_id] = temp[self.mutation_table_id].astype(int)
                chunk_mutations = pd.concat([chunk_mutations, temp])
            chunk_mutations = chunk_mutations.set_index(self.mutation_table_id)
            mutations = mutations.merge(chunk_mutations, how='left', left_index=True, right_index=True)
        if not quiet: print "Done"
        return mutations


    def load_features(self, quiet=None, override=False, by=None, feature_categories=None):
        # Allow 'by' shortcuts to be consistent with the load function()
        if by == 'donor':    by = self.donor_id_column
        if by == 'mutation': by = self.mutation_id_column

        quiet = self.quiet if quiet==None else quiet
        ## db_uri
        if self.db_uri==None:
            print "Plese select a database using the set_backing_database(database_uri) method."
            return
        # Check to see if this has been done before
        if self.loaded and not override:
            print "The MutationMatrix features have already been loaded. Use 'override' set to 'True' to load features again."
            return
        # Check to see that 'by' is either a valid column or None (to just chunk data for encoding as done by _encode_data)
        if by not in list(self.columns) + [None]:
            print "Please choose a valid column to encode by, or in other words, which column you want to group mutations by while encoding. Use 'None' to encode the MutationMatrix index (no grouping)."
            return

        # The final set of encoded mutations 
        loaded_features = None
        feature_categories = feature_categories or self.db_metadata['features'].keys()
        if by!=None:
            item_list = set(self[by])
            # For each item
            for i, item in enumerate(item_list):
                item_mutations = pd.DataFrame(self[self[by]==item])
                if not quiet: print "[%d mutations]" % item_mutations.shape[0]

                # Add mutations features to final dataframe
                loaded_features = pd.concat([loaded_features, self._load_feature_data(item_mutations, quiet=quiet, feature_categories=feature_categories)])
        else:
            mutations = pd.DataFrame(self)
            # Add encoded mutations to final dataframe
            loaded_features = self._load_feature_data(mutations, quiet=quiet, feature_categories=feature_categories)

        # Finalize data and store it back to the MutationMatrix
        self._data = loaded_features._data

        # The table columns have updated, so need to reset the features
        self.feature_categories = feature_categories
        self.set_features()
        self.loaded = True
        return


    # strategies: a dictionary mapping feature (table) name to defined scikitlearn.preprocessing or category_encoders encoder class with a fit_transform() function
    # sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    # category_encoders: http://contrib.scikit-learn.org/categorical-encoding/index.html 
    # to_numeric columns are ignored, categorical encoding defaults to defaults to 'one-hot'
    def encode(self, strategies={}, quiet=None, override=False, features=None, drop_undefined=True):
        quiet = self.quiet if quiet==None else quiet
        ## db_uri
        if self.db_uri==None:
            print "Plese select a database using the set_backing_database(database_uri) method."
            return
        # Check to see if this has been done before
        if self.encoded and not override:
            print "The MutationMatrix features have already been encoded. Use 'override' set to 'True' to encoded features again."
            return

        final_encoded = pd.DataFrame(self)
        features = features or self.feature_categories
        for i, tbl in enumerate(features):
            if not quiet:
                try:
                    clear_output(wait=True)
                except:
                    pass
                print "%2d/%2d [%3.1f%% complete]  Encoding feature '%s'  %s \r" % (i+1, len(features), 100*((i+1)/float(len(features))), tbl, " "*100)
                sys.stdout.flush()

            # Get the (sub)features of table
            subfeatures = self.db_metadata['features'][tbl]
            # Many features will not have any subfeatures (>1 column) in their feature table, make a singleton list of (sub)features
            if type(subfeatures) != list:
                subfeatures = [subfeatures]
            # For each (sub)feature, determine its type
            for subfeature in subfeatures:
                column = subfeature['column'].lower()
                tbl_type = subfeature['type']
                # If categorical, encode the feature with the given strategy, or use a dummy encoding
                if tbl_type=='category':
                    strat   = 'one-hot'
                    encoder = None
                    encoded = None
                    if tbl in strategies.keys():
                        strat = strategies[tbl]
                    # Use CE encoders for the following
                    if strat in ['one-hot', 'binary']:
                        if strat == 'one-hot':
                            encoder = ce.OneHotEncoder()
                            encoded = encoder.fit_transform(list(final_encoded[column].fillna('undefined')))
                            # Relabel the columns
                            ce_map = dict(encoder.ordinal_encoder.mapping[0]['mapping'])
                            rev_map = dict([(b,a) for a,b in ce_map.items()])
                            new_cols = [rev_map[int(a.split('_')[-1])] for a in list(encoded.columns)]
                            new_cols =  ["%s|%s" % (column, a.replace(' ','_').lower()) for a in new_cols]
                            encoded.columns = new_cols
                            if drop_undefined:
                                encoded = encoded.drop("%s|undefined" % (column), axis=1)
                        elif strat == 'binary':
                            encoder = ce.BinaryEncoder()
                            encoded = encoder.fit_transform(list(final_encoded[column].fillna('undefined')))
                            encoded.columns = [a.replace('0_',column+'|bit',1) for a in encoded.columns]
                            if drop_undefined:
                                encoded = encoded.drop("%s|undefined" % (column), axis=1)
                        encoded.index = final_encoded.index
                    # Otherwise default to scikit-learn encoders
                    else:
                        if strat == 'label':
                            encoder = LabelEncoder()
                            encoded = pd.Series(encoder.fit_transform(final_encoded[column]), name=column, index=final_encoded.index)
                        if strat == 'rarity':
                            encoder = dict(final_encoded[column].value_counts()/len(final_encoded[column]))
                            encoder[np.nan] = 0
                            encoded = final_encoded[column].apply(lambda k: encoder[k])
                    # Remove the old data and update with the encoded data
                    final_encoded.drop(column, 1, inplace=True)
                    final_encoded = pd.concat([final_encoded, encoded], axis=1)
                    # Store the encoder for possible future encoding
                    self.feature_transforms[column] = encoder
        self._data = final_encoded._data
        self.set_features()
        self.encoded = True
        if not quiet: print "Done"
        return


    def collapse(self, quiet=None, override=False, by='donor_id', how='mean'):
        quiet = self.quiet if quiet==None else quiet
        # Check to see that 'by' is a valid column
        if by not in list(self.columns)+[self.index.name]:
            print "Please choose a valid column to collapse by. Use [MutationMatrix].columns to see valid options"
            return
        
        if how not in ['mean','median']:
            print "Please choose a valid way to collapse data ('mean', 'median')."
            return

        # Check to see if this has been done before
        if self.collapsed and not override:
            print "This MutationMatrix has already been collapsed. Use 'override' set to 'True' to collapse again."
            return

        # Get a list of unique items in the collapse column and iterate over them
        collapsed_mutations = None
        if by==self.index.name:
            item_list = list(set(self.index))
        else:
            item_list = list(set(self[by]))
        
        for i, item in enumerate(item_list):
            if not quiet: 
                try:
                    clear_output(wait=True)
                except:
                    pass
                print "%2d/%2d [%3.1f%% complete]  Collapsing '%s'  %s \r" % (i+1, len(item_list), 100*((i+1)/float(len(item_list))), item, " "*100)
                sys.stdout.flush()
            item_mutations = None
            # Subset the current items data
            if by==self.index.name:
                item_mutations = self[self.index==item]
            else:
                item_mutations = self[self[by]==item]
            # Skip if only one value
            if not item_mutations.shape[0]<2:
                # Otherwise get feature data
                temp = item_mutations[list([by]+list(self.features))].apply(pd.to_numeric, errors='ignore')
                # Group by donor id and calculate using 'how'
                if how=='mean':
                    temp = temp.groupby(by).mean()
                elif how=='median':
                    temp = temp.groupby(by).median()
                # Add the clean collapsed donor mutations to the final dataframe
                collapsed_mutations = pd.concat([collapsed_mutations, temp])
            # For only one mutation
            else:
                item_mutations = item_mutations.set_index(by)
                collapsed_mutations = pd.concat([collapsed_mutations, item_mutations])

        self._data = collapsed_mutations._data
        self.collapsed = True
        if not quiet: print "Done"
        return


    def select_features(self, model=None, n_splits=50, quiet=None):
        quiet = self.quiet if quiet==None else quiet
        if model==None:
            print "No model specified, using a random forest classifier"
            model =  RandomForestClassifier(n_estimators=50)


        X, Y = self.normalize2XY()
        features = self.features
        tissue_map = {a: b for a,b in enumerate(Y.unique())}
        map_tissue = {b: a for a,b in tissue_map.iteritems()}
        Y = np.array(Y.apply(lambda k: map_tissue[k]))
        X = np.array(X)
        scores = defaultdict(list)
        
        n = 1
        for train_idx, test_idx in ShuffleSplit(n_splits=n_splits, test_size=.25).split(X):
            if not quiet: 
                try:
                    clear_output(wait=True)
                except:
                    pass
                print "%2d/%2d [%3.1f%% complete]  Split %d\r" % (n, n_splits, 100*((n)/float(n_splits)), n)
                sys.stdout.flush()
                n+=1
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = model.fit(np.array(X_train), np.array(Y_train))
            acc = r2_score(Y_test, model.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, model.predict(X_t))
                scores[features[i]].append((acc-shuff_acc)/acc)
        new_features = sorted([(round(np.mean(score),4), feat) for feat, score in scores.items()], reverse=True)
        if not quiet: print "Done"
        return pd.DataFrame(new_features, columns=['Score', 'Feature'])


    def normalize2XY(self, normalize_options=None, quiet=None):
        # Set defaults
        normalize_options = self.normalize_options if normalize_options == None else normalize_options
        quiet             = self.quiet             if quiet             == None else quiet

        Y = None
        if self.label_column:
            Y = pd.Series(self[self.label_column])
        else:
            print "You should specify a label column before calling normalize_XY"
            return
        X = self[self.features]
        X, imputer, scaler = self._normalize(X)
        X = pd.DataFrame(X)
        X.columns = self.features
        return (X,Y)


    def normalize(self, normalize_options=None, quiet=None, override=False):
        # Set defaults
        normalize_options = self.normalize_options if normalize_options == None else normalize_options
        quiet             = self.quiet             if quiet             == None else quiet

        if override or not self.normalized:
            labels = None
            if self.label_column:
                labels = self[self.label_column]
            else:
                print "You should specify a label column before calling normalize_XY"
                return
            X = self[self.features]
            X, imputer, scaler = self._normalize(X)
            self._data      = pd.DataFrame(X)._data
            self.columns    = self.features
            self._data      = pd.concat([pd.DataFrame(list(labels), columns=['label']), self], axis=1)._data
            self.imputer    = imputer
            self.scaler     = scaler
        else:
            print "This MutationMatrix has already been normalized. Use 'override' set to 'True' to normalize again."
        
        self.normalized = True
        return


    def random_forest(self, optimize_parameters=False, estimate_stability=True, sanity_check=True, cv=5, n_estimators=40, test_size=0.30, model_parameters=None):
        if self.label_column==None:
            print "You should specify a label column with set_label_column() before running the random forest"
            return
        print "Running random forest..."
        
        # Normalize data
        X = self[self.features]
        X, self.imputer, self.scaler = self._normalize(X)
        Y = list(self[self.label_column].apply(str))

        # Split the dataset in testing and training sets and create a vanilla random forest classifer as the model
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        self.model = RandomForestClassifier(n_estimators=n_estimators)

        # If grid search is requested, do it, and update the model with the best parameters
        if optimize_parameters:
            print "Optimizing parameters using training data..."
            sys.stdout.flush()
            param_grid = {
                          "max_depth": [1, 3, 5, None],
                          "max_features": [1, .5, 10, 'auto', None],
                          "min_samples_split": [2, 3, 10],
                          "min_samples_leaf": [2, 3, 10],
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]
                         }
            grid_search = self._grid_search(param_grid)
            model_parameters = grid_search.best_params_
        
        # If available re-instantiate the model with either passed model_parameters generated or those generated with optimize_parameters=True
        if model_parameters is not None:
            self.model = RandomForestClassifier(n_estimators=n_estimators, **model_parameters)
        
        # Use One-vs-rest classification scheme for multi-class case
        if len(set(Y))>2:
            print "Multiclass prediction detected, using a one-vs-rest classification strategy"
            sys.stdout.flush()
            self.model = OneVsRestClassifier(self.model)

        print "Modeling with training data"
        self.model.fit(self.X_train, self.Y_train)

        if estimate_stability:
            print "Estimating model stability with %d-fold cross validation" % cv
            sys.stdout.flush()
            self._cross_validate(cv=cv)

        if sanity_check:
            print "Running sanity check by modeling with shuffled labels..."
            sys.stdout.flush()
            self._sanity_check(cv=cv)


    def svm(self, optimize_parameters=False, estimate_stability=False, sanity_check=False, cv=5, n_estimators=40, test_size=0.30, model_parameters=None):
        if self.label_column==None:
            print "You should specify a label column with set_label_column() before running the SVM"
            return
        print "Running support vector machine..."
        
        # Normalize data
        X = self[self.features]
        X, self.imputer, self.scaler = self._normalize(X)
        Y = list(self[self.label_column])

        # Split the dataset in testing and training sets and create a vanilla random forest classifer as the model
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        self.model = svm.SVC(kernel='linear', probability=True)

        # If grid search is requested, do it, and update the model with the best parameters
        if optimize_parameters:
            print "Optimizing parameters not yet supported for the SVM..."
            ##sys.stdout.flush()
            ##param_grid = {
            ##              "max_depth": [1, 3, 5, None],
            ##              "max_features": [1, .5, 10, 'auto', None],
            ##              "min_samples_split": [2, 3, 10],
            ##              "min_samples_leaf": [2, 3, 10],
            ##              "bootstrap": [True, False],
            ##              "criterion": ["gini", "entropy"]
            ##             }
            ##grid_search = self._grid_search(param_grid)
            ##model_parameters = grid_search.best_params_
        
        # If available re-instantiate the model with either passed model_parameters generated or those generated with optimize_parameters=True
        if model_parameters is not None:
            self.model = svm.SVC(**model_parameters)
        
        # Use One-vs-rest classification scheme for multi-class case
        if len(set(Y))>2:
            print "Multiclass prediction detected, using a one-vs-rest classification strategy"
            sys.stdout.flush()
            self.model = OneVsRestClassifier(self.model)

        print "Modeling with training data"
        self.model.fit(self.X_train, self.Y_train)

        if estimate_stability:
            print "Estimating model stability with %d-fold cross validation" % cv
            sys.stdout.flush()
            self._cross_validate(cv=cv)

        if sanity_check:
            print "Running sanity check by modeling with shuffled labels..."
            sys.stdout.flush()
            self._sanity_check(cv=cv)


    def predict(self, X_df=None):
        # Normalize data
        X = None
        if X_df is None:
            X = self.X_test
        else:
            # Create missing columns and fill with NaN
            missing = pd.DataFrame([], columns=set(self.features).difference(set(X_df.columns)))
            X_df = pd.concat([X_df, missing], axis=1)
            X = X_df[self.features]
            X = self._normalize(X, imputer=self.imputer, scaler=self.scaler)[0]
        
        probabilities = pd.DataFrame(self.model.predict_proba(X))
        probabilities.columns = [str(a) for a in list(self.model.classes_)]
        probabilities = probabilities.reset_index(drop=True)
        if X_df is None:
            predictions = pd.concat([pd.Series(self.Y_test), probabilities], axis=1)
            predictions.columns = ["Actual"] + list(predictions.columns[1:])
        else:
            X_df = X_df.reset_index(drop=True)
            predictions = pd.concat([X_df, probabilities], axis=1)
        return predictions

    def _get_colors(self, cmap="Reds", n=8):
        cmap = cm.get_cmap(cmap, n)
        colors = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        return colors


    def show_dendrogram(self, color_labels=True, heat_cmap="Oranges", label_cmap="bone", downsample_n=None):
        if not(self.shape[0]<1000 or (downsample_n and downsample_n<1000)):
            print "Attempting to show a dendrogram for more than 1,000 entries is currently disabled. You can try using the 'downsample_n' parameter with a given integer n < 1,000 to randomly sample n entries."
            return

        Xd = None
        if downsample_n:
            Xd = self.sample(downsample_n)
        else:
            Xd = self
        Yd = list(Xd[self.label_column])
        Xd = Xd[self.features]
        Xd = self._normalize(Xd)[0]
        hm = pd.DataFrame(Xd)
        hm.columns = self.features
        hm.index = Yd

        figsize=(min(100, max(int(.5*hm.shape[1]),3)), min(100, max(int(.5*hm.shape[0]),3)))

        # Assign color codes to each label
        label_colors = self._get_colors(cmap=label_cmap, n=len(set(Yd)))
        label_colors = [hex2color(a) for a in label_colors]
        label_colors = dict(zip(map(str, set(Yd)), label_colors))
        label_colors = pd.Series(Yd).apply(str).map(label_colors)

        cg = sns.clustermap(hm, method='complete', cmap=heat_cmap, linewidths=.05, figsize=figsize, row_colors=label_colors if color_labels else None) # yticklabels=2 to skip every other label
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
        plt.setp(cg.ax_heatmap.get_xticklabels(), size=20)
        plt.setp(cg.ax_heatmap.get_yticklabels(), size=20)
        cg.cax.set_visible(False)


    def show_class_predictions(self, as_heatmap=False, cmap="Reds"):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        labels = pd.DataFrame(self.Y_test)
        predictions = pd.DataFrame(self.model.predict(self.X_test))
        probabilities = pd.DataFrame(self.model.predict_proba(self.X_test))
        predictions = pd.concat([labels, predictions, probabilities], axis=1)
        predictions.columns = [["Actual", "Predicted"]+list(self.model.classes_)]


        if as_heatmap:
            sns.set_style("white")
            sns.set_context("poster")
            dim = min(100, max(int(.5*predictions.shape[0]),3))
            figsize=(dim,dim)
            fig, ax = plt.subplots()
            # the size of A4 paper
            fig.set_size_inches(figsize)
            shown = predictions.drop(["Predicted"], axis=1).set_index('Actual')
            ax = sns.heatmap(shown, linewidths=.5, square=True, cmap=cmap)
            ax.set_xlabel("Prediction Probability",fontsize=20)
            ax.set_ylabel("Actual",fontsize=20)
            ax.tick_params(labelsize=12)
        return predictions


    def print_report(self):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        confusions = self.show_confusion_matrix(as_heatmap=False, normalize=False, return_data=True)
        FP = confusions.sum(axis=0) - np.diag(confusions)
        FN = confusions.sum(axis=1) - np.diag(confusions)
        TP = pd.Series(np.diag(confusions), dtype='int64', )
        TP.index = FN.index
        TN = confusions.values.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        report = pd.DataFrame([FP,FN,TP,TN,TPR,FPR,PPV,NPV,FPR,FNR,FDR,ACC]).transpose()
        report.columns = ['FP','FN','TP','TN','TPR','FPR','PPV','NPV','FPR','FNR','FDR','ACC']
        
        # Add means
        means = pd.DataFrame(report.mean()).transpose()
        means.index = ['Mean']
        report = pd.concat([report,means], axis=0)
        return report


    def show_confusion_matrix(self, as_heatmap=True, cmap="Reds", annot=True, font_size=20, normalize=True, return_data=False):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        labels = pd.DataFrame(self.Y_test)
        predictions = pd.DataFrame(self.model.predict(self.X_test))
        probabilities = pd.DataFrame(self.model.predict_proba(self.X_test))
        predictions = pd.concat([labels, predictions, probabilities], axis=1)
        predictions.columns = [["Actual", "Predicted"]+list(self.model.classes_)]

        confusion = pd.DataFrame(confusion_matrix(predictions['Actual'], predictions['Predicted']))
        confusion.columns = sorted(list(set(predictions['Actual'])))
        confusion.index   = confusion.columns
        #confusion = pd.DataFrame()
        #classes = sorted(list(set(predictions['Actual'])))
        #for tA in classes:
        #    for tB in classes:
        #        t = predictions[(predictions['Actual']==tA)]
        #        n = t[t['Predicted']==tB].shape[0]
        #        confusion.ix[tA,tB] = n

        if normalize:
            confusion = confusion.div(confusion.sum(axis=1), axis=0)
        if as_heatmap:
            plt.figure(figsize=(15, 10))
            fmt=".2f" if normalize else "0.0f"
            sns.heatmap(confusion, cmap=cmap, annot=annot, fmt=fmt, linewidths=.5, annot_kws={"size": font_size})   
        return confusion 


    def print_mean_confidences(self):
        labels = pd.DataFrame(self.Y_test)
        probabilities = pd.DataFrame(self.model.predict_proba(self.X_test))
        probabilities.columns = list(self.model.classes_)
        confidences = pd.DataFrame((abs(probabilities-(1/len(labels)))+(1/len(labels))).mean())
        confidences.columns = ['Mean Confidence']
        return confidences


    def classify_predictions(self, include_training=False):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        labels = None
        X = None
        if include_training:
            labels = pd.DataFrame(self.Y_train + self.Y_test)
            X = pd.concat([pd.DataFrame(self.X_train), pd.DataFrame(self.X_test)])
        else:
            labels = pd.DataFrame(self.Y_test)
            X = pd.DataFrame(self.X_test)

        classes       = list(self.model.classes_)
        predictions   = pd.DataFrame(self.model.predict(X))
        probabilities = pd.DataFrame(self.model.predict_proba(X))
        predictions   = pd.concat([labels, predictions, probabilities], axis=1)
        predictions.columns = [["Actual", "Predicted"]+classes]
        predictions.columns = [str(int(a)) if type(a) in [np.int64, np.float64] else a for a in predictions.columns]

        # Avoid numeric labels
        predictions.columns = [str(int(a)) if type(a) in [np.int64, np.float64] else a for a in predictions.columns]
        if predictions['Actual'].dtype in [np.int64, np.float64]:
            predictions['Actual']=predictions['Actual'].apply(lambda k: str(int(k)))
        if predictions['Predicted'].dtype in [np.int64, np.float64]:
            predictions['Predicted']=predictions['Predicted'].apply(lambda k: str(int(k)))

        confidence_probs = []
        for i,j in predictions.iterrows():
            # The positive case 
            confidence_probs.append({'Type': 'Positive', 'Class': j['Actual'], 'Probability': j[j['Actual']]})
            # The negative case 
            for t in set(classes)-set(j['Actual']):
                confidence_probs.append({'Type': 'Negative', 'Class': t, 'Probability': j[t]})
            # The true positive case
            if j['Actual']==j['Predicted']:
                confidence_probs.append({'Type': 'True Positive', 'Class': j['Actual'], 'Probability': j[j['Actual']]}) 
            else:
                # The false positive case for the predicted class
                confidence_probs.append({'Type': 'False Positive', 'Class': j['Predicted'], 'Probability': j[j['Actual']]})
                # The false negative case for the actual class
                confidence_probs.append({'Type': 'False Negative', 'Class': j['Actual'], 'Probability': j[j['Actual']]})
                # The true negative for all but actual class
                for t in set(classes)-set((j['Actual'], j['Predicted'])):
                    confidence_probs.append({'Type': 'Negative', 'Class': t, 'Probability': j[t]})
        # Convert to dataframe
        confidence_probs = pd.DataFrame(confidence_probs)
        return confidence_probs



    def show_confidence_plot(self, left="True Positive", right="False Positive", left_color="#EEEEEE", right_color="#AACFE5", plot_options=None, title='Classification Probability Distributions'):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        # Allow passed plot options to superceed default options
        default_options = {
                            'split':      True, 
                            'inner':      'quart',  # or stick
                            'trim':       True,
                            'cut':        0,
                            'saturation': 1,
                            'linewidth':  0.8,
                            'scale':      'width'
                        }
        if plot_options != None:
            for k,v in plot_options.items():
                if k in default_options.keys():
                    default_options[k]=v
        plot_options = default_options

        confidence_probs = self.classify_predictions()
        confidence_probs_plotted = confidence_probs[(confidence_probs['Type']==left) | (confidence_probs['Type']==right)]
        confidence_probs_plotted = confidence_probs_plotted.sort_values(by=['Class','Type'], ascending=[True, False])

        sns.set(style="white")
        sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2})
        plt.figure(figsize=(20, 15))
        g = sns.violinplot(
                        x='Class', 
                        y='Probability', 
                        hue='Type', 
                        data=confidence_probs_plotted, 
                        palette=[left_color, right_color],
                        **plot_options
                      )
        sns.despine()
        plt.xticks(rotation=90)
        _ = plt.ylim(0,1.1)
        plt.xlabel('')
        plt.ylabel('Probability Distribution')
        plt.title(title)
        plt.legend(bbox_to_anchor=(0.25, 0.98), loc="bottom", borderaxespad=0.)



    def my_pr_curves(self, title='', cmap="terrain"):
        # Generate predictions 
        predictions=self.predict()
        predictions = predictions.set_index('Actual')
        columns = list(set(predictions.columns))

        # Setup the plot
        colors = self._get_colors(cmap=cmap, n=len(columns))
        colormap = dict(zip(columns, colors))
        sns.set(style="white")
        sns.set_context("poster", font_scale=2, rc={"lines.linewidth": 2})
        plt.figure(figsize=(30, 20))

        # Calculate PR curves for each column

        for column in columns:
            x = [0.]
            y = [1.]
            auc = 0
            other_columns = [t for t in columns if t!=column]
            subset = predictions.ix[predictions.index==column]
            this   = subset[column]
            others = subset[other_columns]
            for i in xrange(0,101,1):
                cutoff = (i/100.)
                TP = (this>=cutoff).sum()
                FP = ((others>=cutoff).sum()).mean()
                AP = len(this) 
                PP = TP+FP
                PR = TP/float(PP)
                RC = TP/float(AP)
                if np.isnan(PR):
                    PR = 1.0
                x.append(PR)
                y.append(RC)
                auc += RC*PR
            auc /= 100
            #x = np.maximum.accumulate(x[::-1])[::-1]
            final = pd.DataFrame({'x':x, 'y':y})
            final = final.sort_values(by='x')
            plt.plot(final['x'], final['y'], color=colormap[column], alpha=.75, lw=6, label='{0:} ({1:0.2f})'.format(column, auc))
        plt.plot([0,1], [0.5,0.5], 'k--', lw=1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.001, 1.05])
        plt.title(title)
        sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2})
        plt.show()
        return

    def show_roc_curves(self, title='ROC Curve(s)', *args, **kwargs):
        show_curves(*args, metric='ROC', title=title, **kwargs)
    def show_pr_curves(self, title='Precision Recall Curve(s)', *args, **kwargs):
        show_curves(*args, metric='PR', **kwargs)

    def show_curves(self, metric='ROC', title='', cmap="terrain", return_data=False, styles=None, class_order=None):
        if self.model==None:
            print "You should run a model before plotting curves"
            return

        classes = list(self.model.classes_)
        try:
            n_classes = self.model.n_classes_
        except AttributeError:
            n_classes = len(self.model.classes_)
        try:
            # Will throw an exception for multiclass scenarios
            binn = self.model.label_binarizer_.set_params(sparse_output=False)
            y_test = binn.transform(self.Y_test)
            classes = binn.classes_
        except AttributeError:
            label_map = dict([(v,k) for k,v in enumerate(self.model.classes_)])
            y_test = np.array(map(lambda k: label_map[k], self.Y_test))[:, np.newaxis]

        if class_order:
            if len(set(class_order).difference(set(classes)))!=0:
                print "Please only provide classes that exist in the model when specifying class_order"
                return
            classes = class_order

        y_score = self.model.predict_proba(self.X_test)

        # label_binarizer will produce a single column label for binary classifications when transforming Y_test,
        # however predict_proba produces a two column probability matrix when predicting X_test. To fix this issue, 
        # convert y_test to a two column probability matrix by taking the probability complement. 
        if n_classes<=2:
            y_test = np.hstack((1-y_test, y_test))

        # Compute ROC curve and ROC area for each class
        x = dict()
        y = dict()
        curve_aucs = dict()
        #calls = np.argmax(y_score, axis=1)
        #(calls==i).astype(int)
        for i in range(n_classes):
            this_class = classes[i]
            if metric=='ROC':
                x[this_class], y[this_class], _ = roc_curve(y_test[:, i], y_score[:, i], drop_intermediate=True)
                curve_aucs[this_class] = auc(x[this_class], y[this_class])
                # Start drawing curves at [0,0]
                x[this_class] = np.concatenate([[0],x[this_class]])
                y[this_class] = np.concatenate([[0],y[this_class]])
            else: #metric=='PR'
                x[this_class], y[this_class], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
                #x[this_class], y[this_class], a, b = precision_recall_fscore_support(y_test[:, i], y_score[:, i], average='samples')
                curve_aucs[this_class] = average_precision_score(y_test[:, i], y_score[:, i])
                # Start drawing curves at [0,1]
                x[this_class] = np.concatenate([[0],x[this_class]])
                y[this_class] = np.concatenate([[1],y[this_class]])


        # Get micro/macro averages for the multiclass case
        if n_classes>2:
            if metric=='ROC':
                # Compute micro-average ROC curve and ROC area
                # See http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
                x["micro"], y["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                curve_aucs["micro"] = auc(x["micro"], y["micro"])

            if metric=='PR':
                x["micro"], y["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
                curve_aucs["micro"] = average_precision_score(y_test, y_score, average="micro")
            
            # Compute macro-average curve and area
            # First aggregate all false positive rates
            all_x = np.unique(np.concatenate([x[i] for i in classes]))

            # Then interpolate all ROC curves at this points
            mean_y = np.zeros_like(all_x)
            for i in classes:
                mean_y += interp(all_x, x[i], y[i])

            # Finally average it and compute AUC
            mean_y /= n_classes

            x["macro"] = all_x
            y["macro"] = mean_y
            curve_aucs["macro"] = auc(x["macro"], y["macro"])



        # Plot all ROC curves
        #colors = sns.color_palette("hls", n_classes)
        #colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        colors = self._get_colors(cmap=cmap, n=n_classes)
        sns.set(style="white")
        sns.set_context("poster", font_scale=2, rc={"lines.linewidth": 2})
        plt.figure(figsize=(30, 20))
        #plt.plot(x["micro"], y["micro"],
        #         label='micro-average ({0:0.2f})'
        #               ''.format(curve_aucs["micro"]),
        #         color='deeppink', linestyle='-', linewidth=4)

        if n_classes>2:
            try:
                plt.plot(x['macro'], y['macro'], label='macro-average ({0:0.2f})'.format(curve_aucs['macro']), **styles['macro'])
            except:
                plt.plot(x['macro'], y['macro'], label='macro-average ({0:0.2f})'.format(curve_aucs['macro']), color='k', linestyle='-.', linewidth=4)

            for this_class, color in zip(classes, colors):
                if color=="#ffffff": color="#dddddd"
                try:
                    plt.plot(x[this_class], y[this_class], label='{0:} ({1:0.2f})'.format(this_class, curve_aucs[this_class]), **styles[this_class])
                except:
                    plt.plot(x[this_class], y[this_class], color=color, alpha=.75, lw=6, label='{0:} ({1:0.2f})'.format(this_class, curve_aucs[this_class]))
        
        # For two class case, only one curve needs to be shown
        else:
            # Pick the first class to show, or show the 'True' class if it exists
            this_class = classes[0]
            if any([a==True for a in classes]):
                this_class = True
            # Plot with given styles, or use the default
            try:
                plt.plot(x[this_class], y[this_class], label='{0:} ({1:0.2f})'.format(this_class, curve_aucs[this_class]), **styles[this_class])
            except:
                plt.plot(x[this_class], y[this_class], color=colors[0], alpha=.75, lw=6, label='{0:} ({1:0.2f})'.format(this_class, curve_aucs[this_class]))

        # Add lables and clean up the graph
        if metric=='ROC':
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
        elif metric=='PR':
            plt.plot([0,1], [0.5,0.5], 'k--', lw=1)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="lower left")
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.001, 1.05])
        plt.title(title)
        sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2})
        plt.show()

        if return_data:
            return {'xs':x, 'ys':y, 'aucs':curve_aucs}


    def show_feature_importances(self, how="boxplot", return_data=False, order=False):
        if self.model==None:
            print "You should run a model before making class predictions"
            return

        fws = None
        if type(self.model)==OneVsRestClassifier:
            for est in self.model.estimators_:
                try:
                    fws = pd.concat([fws, pd.DataFrame(est.feature_importances_).transpose()], axis=0)
                except AttributeError:
                    fws = pd.concat([fws, pd.DataFrame(est.coef_)], axis=0)
            fws.columns = self.features
            fws.index = self.model.classes_
        else:
            try:  #RF
                imp = pd.DataFrame(self.model.feature_importances_)
                fws = pd.concat([imp, imp], axis=1).transpose()
            except AttributeError:  # SVM
                imp = pd.DataFrame(self.model.coef_)
                fws = pd.concat([imp, imp], axis=0)
            fws.columns = self.features
            fws.index = self.model.classes_
            how = 'barplot' if how=='boxplot' else how

        if how=="table":
            return fws

        stack = pd.DataFrame(fws.stack())
        stack.columns=['Importance Value']
        stack['Feature'] = [s[1] for s in stack.index]
        stack['Class']  = [s[0] for s in stack.index]

        # Order by importance if requested
        if order==True:
            stack = stack.sort_values(by='Importance Value', ascending=False)
        else:
            stack = stack.sort_index()

        # Plot
        sns.set(style="dark")
        sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2})

        colors = sns.cubehelix_palette(rot=-.3, light=1)

        sns.set_style("white")
        #sns.set_style("whitegrid", {
        #    "ytick.major.size": 1.0,
        #    "xtick.major.size": 1.0,
        #    'grid.linestyle': '--'
        # })

        fig, ax = plt.subplots()
        fig.set_size_inches(45, 10)
        if how=="byclass":
            sns.stripplot(x='Feature', y='Importance Value', hue='Class', data=stack, jitter=False, alpha=.85, size=10)
        elif how=="barplot":
            sns.barplot(x='Feature', y='Importance Value', data=stack)
        else: #how=="boxplot"
            sns.boxplot(x='Feature', y='Importance Value', data=stack)
        plt.title("Model Feature Importances", fontsize=20)
        plt.xticks(rotation=90, fontsize=12)
        sns.despine()

        if return_data:
            return stack


    def save_matrix(self, filename="MutationMatrix.pkl"):
        joblib.dump(self, filename)

    def save_model(self,  filename="MutationMatrixModel.pkl"):
        with open(filename, 'wb') as f:
            dill.dump([
                        self.features,
                        self.label_column,
                        self.imputer,
                        self.scaler,
                        self.model,
                        self.normalize_options,
                      ], f)

    def load_model(self,  filename="MutationMatrixModel.pkl"):
        with open(filename, 'rb') as f:
            data = dill.load(f)
            self.features             = data[0]
            self.label_column         = data[1]
            self.imputer              = data[2]
            self.scaler               = data[3]
            self.model                = data[4]
            self.normalize_options    = data[5]


    def _normalize(self, X, normalize_options=None, quiet=None, imputer=None, scaler=None):
        # Set defaults
        normalize_options = self.normalize_options if normalize_options == None else normalize_options
        quiet             = self.quiet             if quiet             == None else quiet

        # Get input options
        nan_strat    = normalize_options['nan_strat']
        scaler_strat = normalize_options['scaler_strat']

        # Check inputs
        if nan_strat not in ('zero','mean','median', 'most_frequent'):
            print "Unknown NaN imputation strategy, please use 'zero', 'mean', 'median', or 'most_frequent'"
            return
        if scaler_strat not in ('mms', 'standard'):
            print "Unknown scaler strategy, please use 'mms' or 'standard' (z-score)"
            return 


        # Issues with nans between Pandas DataFrame and Numpy array, but this should correct problems downstream
        X = np.array(pd.DataFrame(X).apply(pd.to_numeric, errors='coerce'))

        # Deal with all zero columns
        if not quiet: print "Normalizing data..."
        if not quiet: print "Columns with all NaN values will be set to 0."
        X[:, np.all(pd.isnull(X), axis=0)] = 0

        # Deal with remaining zeros
        if not quiet: print "Imputing remaining NaNs using the '%s' strategy." % nan_strat
        if imputer == None:
            if nan_strat=='zero':
                imputer = np.nan_to_num
                X = np.nan_to_num(X)
            else: 
                imputer = Imputer(missing_values='NaN', strategy=nan_strat, axis=0)
                X = imputer.fit_transform(X)
        elif imputer == np.nan_to_num:
            X = np.nan_to_num(X)
        else:
            X = imputer.transform(X)

        # Scale values
        if not quiet: print "Scaling values with the '%s' strategy." % scaler_strat
        if scaler == None:
            if scaler_strat =='mms':
                scaler = MinMaxScaler().fit(X)
            elif scaler_strat=='standard':
                scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        return (X, imputer, scaler)


    def _grid_search(self, param_grid=None):
        if param_grid==None:
            print "To use grid search, please provide a dictionary of possible parameter values."
            return
        if self.X_train==None or self.Y_train==None:
            print "Training data must be specified before running this function. Try calling a modeling function first."
            return 

        # Utility function to report best scores
        def report(results, n_top=3):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    print("Model with rank: {0}".format(i))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                          results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
                    print("Parameters: {0}".format(results['params'][candidate]))
                    print("")

        # run grid search
        grid_search = GridSearchCV(self.model, param_grid)
        start = time.time()
        grid_search.fit(self.X_train, self.Y_train)
        stop = time.time()

        print("GridSearchCV took %.2f seconds." % (stop-start))
        print grid_search.best_params_
        print grid_search.best_score_
        return grid_search


    # Checks the sanity of a model by randomizing the labels and remodeling
    def _sanity_check(self, cv=5):
        if self.label_column==None:
            print "You should specify a label column and run a model before attempting a sanity check of it"
            return
        if self.model==None:
            print "You should train a model before attempting a sanity check of it"
            return
        model = deepcopy(self.model)
        new_labels = deepcopy(self.Y_train)
        print "Shuffling labels..."
        shuffle(new_labels)

        scores = cross_val_score(model, self.X_train, new_labels, cv=cv)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
        print "Expected by chance: %.2f" % (1./len(set(self.Y_train)))


    def _cross_validate(self, cv=5):
        rf_scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=cv)
        print "Accuracy within training set ONLY: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2)
        print "Expected by chance: %.2f" % (1./len(set(self.Y_train)))


def load_matrix(filename="MutationMatrix.pkl"):
    return joblib.load(filename)


