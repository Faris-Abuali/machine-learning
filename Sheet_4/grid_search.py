from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np


def grid_search(
    func: Callable[..., np.ndarray], param_grid: Dict[str, List[Any]]
) -> Tuple[Dict[str, Any] | None, float]:
    """
    Args:
        func: the objective function to optimize. We want to find the parameters that maximize this function.
        func returns an array of scores of shape (k,) where k is the number of folds

        param_grid: a dictionary of the form {param_name: [param_values]}

    Returns: a tuple (best_params, best_score) where best_params is a dictionary of the best parameters and best_score is the best score the objective function achieved with these parameters.

    Example:
    >>> param_grid = {
    >>>   "c": [1, 10],
    >>>   "sigma": [1, 2, 3]
    >>> }

    >>> grid_search(func, param_grid)
    """
    from itertools import product

    best_params: Dict[str, Any] | None = None
    best_score: float = -float("inf")

    # the * operator unpacks the values of the dictionary into a list of tuples:
    # param_grid = {
    #   "c": [1, 10],
    #   "sigma": [1, 2, 3]
    # }
    # list(product(*param_grid.values())) will produce:
    # [(1, 1), (1, 2), (1, 3), (10, 1), (10, 2), (10, 3)]

    for params in product(*param_grid.values()):
        param_dict: Dict[str, Any] = {k: v for k, v in zip(param_grid.keys(), params)}
        # e.g. param_dict = {"c": 1, "sigma": 2}

        score = func(**param_dict).mean()  # func returns numpy.ndarray
        print(f"params: {param_dict}, score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_params = param_dict

    print(best_params)

    return best_params, best_score
