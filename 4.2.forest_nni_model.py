import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import nni
import logging



LOG = logging.getLogger('nni_catboost')

def load_data():
    data_path = 'https://raw.githubusercontent.com/antbartash/max_temp/master/data/data_features.csv'
    data = pd.read_csv(data_path)
    data['DATE'] = data['DATE'].astype('datetime64[ns]')
    X_train = data.loc[data['DATE'].dt.year <= 2021].drop(columns=['TARGET', 'DATE']).copy()
    y_train = data.loc[data['DATE'].dt.year <= 2021, 'TARGET'].copy()
    X_valid = data.loc[data['DATE'].dt.year == 2022].drop(columns=['TARGET', 'DATE']).copy()
    y_valid = data.loc[data['DATE'].dt.year == 2022, 'TARGET'].copy()
    return X_train, X_valid, y_train, y_valid




def get_default_parameters():
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_leaf': 1,
        'max_features': 1.0,
        'min_impurity_decrease': 0.0,
        'ccp_alpha': 0.0
    }
    return params

    
def get_model(PARAMS):
    rng = np.random.RandomState(42)
    model = RandomForestRegressor(random_state=rng)
    model.set_params(**PARAMS)
    return model


def run(X_train, y_train, model):
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))
    LOG.debug('score: %s', score)
    nni.report_final_result(score)


if __name__ == '__main__':
    X_train, _, y_train, _ = load_data()
    try:
        RECIEVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECIEVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECIEVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, y_train, model)
    except Exception as exception:
        LOG.exception(exception)
        raise