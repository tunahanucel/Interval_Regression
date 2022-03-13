# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from mapie.regression import MapieRegressor 
import matplotlib.pyplot as plt
    
class IntervalRegressor(BaseEstimator):
    """
    This class is created for adjust MapieRegressor to scikit-learn regression design in order to easy deployment. 
    To access MapieRegressor, plase see: 'https://mapie.readthedocs.io/en/latest/index.html'
    MapieRegressor enables to find lower and upper bounds of predictions with an alpha value for selected estimator algorithm.
    For example, for alpha = 0.1, MapieRegressor can find the prediction interval with 90% probability.
    For more information, please visit the MAPIE official website.
    
    Parameters:
    output: str, Preferred output. ['mid', 'lower', 'upper'] 
    base_estimator: original name of the prediction algorithm. For example 'Lasso', 'RandomForestRegressor'
    base_estimator_params: dict, dictionary that include parameters of base_estimator
    mapie_alpha: float, alpha value for MapieRegressor, 
        (1-alpha) = the probability of the actual value within prediction interval
    method: str, learning method for MapieRegressor. IntervalRegressor is capable for 'base' an 'naive' methods for now.
        default 'base'
    cv: int, the number of folds for Cross-Validation during the optimazing MapieRegressor, default 3
    
    Note: It is recommended that doing parameter optimization before using this class.
    """
    def __init__(self, output, base_estimator, base_estimator_params, mapie_alpha=0.1, method='base', cv=3):
        self.output = output
        self.mapie_alpha = mapie_alpha
        self.base_estimator = base_estimator
        self.base_estimator_params = base_estimator_params
        self.method = method
        self.cv = cv
    
    def fit(self, X, y):
        base_estimator_ = self.base_estimator(**self.base_estimator_params)
        mapie = MapieRegressor(base_estimator_, self.method, self.cv)
        self.estimator = mapie.fit(X, y)
        return self.estimator

    def predict(self, X):
        y_pred, y_pis = self.estimator.predict(X, self.mapie_alpha)
        if self.output == 'mid':
            return y_pred
        elif self.output == 'lower':
            return y_pis[:,0,0]
        elif self.output == 'upper':
            return y_pis[:,1,0]  
        


def plot_interval_regression(data, y_label, start_index=0, step_index=100, prediction=True):
    """
    data: dataframe, it must includes lower and upper bounds of prediciton intervals that are named 'lower' and 'upper'. 
    If predictions==True, 'data' must includes a 'predictions' column also.
    y_label: string, name of the y_label in graph.
    start_index: int, starting point to plot a certain part of the predictions, default 0
    step_index: int, step lenght of the graph, default 100.
        For example, in default mode, function plots first 100 predictions.
    prediction: bool, to plot actual prediction values, default True.
    """
    data = data.reset_index(drop=True)
    data_ = data[start_index:start_index+step_index]
    
    fig, ax = plt.subplots(1, figsize=(18,8))
    ax.set_xlabel("index", fontsize=16) 
    ax.set_ylabel("{}".format(y_label), fontsize=16)
    ax.plot(data_.index, data_.actual, 'r.', markersize=10, label=u'Observations')
    ax.plot(data_.lower, 'b--', label=u'Lower-bound')
    ax.plot(data_.upper, 'b--', label=u'Upper-bound')
    ax.fill_between(data_.index, data_.lower, data_.upper, alpha=0.3)
    
    if prediction:
        ax.plot(data_.index, data_.predictions, 'g.', markersize=10, label=u'Predictions')
    
    plt.grid('whitegrid')
    plt.legend(loc='upper left')