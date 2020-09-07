## GridSearchCV

from sklearn.model_selection import GridSearchCV

dual=[True,False]

max_iter=[100,110,120,130,140]

param_grid = dict(dual=dual,max_iter=max_iter)

import time

lr = LogisticRegression(penalty='l2')

grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X, y)
#### Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')

## RandomSearchCV

from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()

random_result = random.fit(X, y)
#### Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
