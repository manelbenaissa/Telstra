"""Process Telstra data."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def read_csv(csv_file, column):
    """Preprocess csv file to remove text data."""
    df = pd.read_csv(csv_file,
                     index_col='id',
                     converters={column: (lambda x: int(x.split()[1]))})
    return df


# Preprocess csv files
event_df = read_csv('data/event_type.csv', 'event_type')
log_df = read_csv('data/log_feature.csv', 'log_feature')
resource_df = read_csv('data/resource_type.csv', 'resource_type')
severity_df = read_csv('data/severity_type.csv', 'severity_type')

test_df = read_csv('data/test.csv', 'location')
train_df = read_csv('data/train.csv', 'location')

df = pd.concat([train_df, test_df])


def print_data():
    """Print data to visualize it."""
    print 'Train df'
    print train_df.head()
    print '\nAll data'
    print df.head()
    print df.tail()
    print '\nSeverity df'
    print severity_df.head()
    print '\n Resource type'
    print resource_df.groupby('id').head()
    print '\n Log type'
    print log_df.groupby('id').head()
    print '\n Event type'
    print event_df.groupby('id').head()


def get_features():
    """Extract features from data."""
    # Create a feature dataframe
    X = pd.DataFrame(index=severity_df.index)
    X['fault_severity'] = train_df['fault_severity']
    X['severity_type'] = severity_df['severity_type']
    X['location'] = df['location']

    # Event
    # Number of events by id
    numevents_df = event_df.groupby('id').count()
    X = pd.merge(X, numevents_df,
                 left_index=True, right_index=True, how='left')
    # Type of event by id
    liste_evts = event_df.event_type.unique()
    for row in event_df.itertuples():
        i = row[0]
        for col in liste_evts:
            if row[1] == col:
                X.loc[i, 'event_%s' % (col)] = 1

    # Log features
    # Type of feature by id
    liste_logf = log_df.log_feature.unique()
    for row in log_df.itertuples():
        i = row[0]
        for col in liste_logf:
            if row[1] == col:
                X.loc[i, 'logf_%s' % (col)] = 1
    # Volume mean
    meanvol_df = log_df.groupby('id').mean()
    mean_df = meanvol_df.drop(['log_feature'], axis=1)
    X = pd.merge(X, mean_df, left_index=True, right_index=True, how='left')

    # Resource : type of resource by id
    liste_resource = resource_df.resource_type.unique()
    for row in log_df.itertuples():
        i = row[0]
        for col in liste_resource:
            if row[1] == col:
                X.loc[i, 'resource_%s' % (col)] = 1

    features = X.columns[1:]
    return(X, features)


def classify():
    """Build a RF classifier to process data and evaluate fault severity."""
    X, features = get_features()

    training_df = pd.merge(train_df, X[features],
                           left_index=True,
                           right_index=True, how='inner').fillna(0)
    get_targetindex = list(training_df.columns).index('fault_severity')
    new_features = list(training_df.columns)
    del new_features[get_targetindex]
    # Separate training set from cross validation set
    training_df['is_train'] = np.random.uniform(0, 1, len(training_df)) <= 0.75
    train = training_df[training_df['is_train'] == True]
    test = training_df[training_df['is_train'] == False]
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
    y, _ = pd.factorize(train['fault_severity'])
    clf.fit(train[new_features], y)

    # Apply classifier to test data
    predicted = clf.predict(test[new_features])
    accuracy = accuracy_score(test['fault_severity'], predicted)
    print 'Mean accuracy score: %s' % (accuracy)
    print clf.predict_proba(test[new_features])[0:10]
    preds = X.fault_severity[clf.predict(test[new_features])]
    print 'preds', preds[0:5]
    print test['fault_severity'].head()

    final_df = pd.merge(test_df, X[features],
                        left_index=True, right_index=True, how='left').fillna(0)
    final_proba = clf.predict_proba(final_df[new_features])
    final_df = pd.DataFrame(final_proba,
                            index=test_df.index,
                            columns=['predict_0', 'predict_1', 'predict_2']
                            )
    print final_df.head()
    final_df.to_csv('data/submission.csv')


classify()
