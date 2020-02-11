import pandas as pd
from surprise import SVD, SVDpp, NMF,SlopeOne,CoClustering,NormalPredictor
from surprise import model_selection


def matrix_factorization_param(data_cv):
    # Iterate over all algorithms
    benchmark = []

    for algorithm in [SVD(), SVDpp(), NMF(),SlopeOne(),NormalPredictor(),CoClustering()]:
        # Perform cross validation
        results = model_selection.cross_validate(algorithm, data_cv,
                                                 measures=['RMSE', 'MAE'], cv=5, verbose=False)
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
    
    rmse = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_mae')
    #print(rmse)

    # Parameter grid
    param_grid = {'n_factors': [100, 150, 200],
                  'n_epochs': [20, 40],
                  'lr_all': [0.001, 0.005, 0.008],
                  'reg_all': [0.075, 0.1, 0.15]
                  }
    algorithm_gs = model_selection.GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
    algorithm_gs.fit(data_cv)

    # best parameters for a model with the lowest rmse
    best_algo = algorithm_gs.best_estimator['rmse']
    return best_algo
