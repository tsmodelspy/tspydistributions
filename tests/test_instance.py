from tspydistributions import pdqr
import tspydistributions.helpers as helpers
import inspect
import numpy as np

def d_instance(names):
    results = []
    for name in names:
        fun = getattr(pdqr, f'd{name}')  
        params = inspect.signature(fun).parameters
        defaults = helpers._default_parameter_values(name)
        args = {"x": 0}
        k = 0
        for param in ['mu', 'sigma', 'skew', 'shape', 'lamda']:
            if param in params:
                args[param] = defaults[k]
                k += 1
        results.append(isinstance(fun(**args), np.ndarray))
    return results

def test_d():
    out = d_instance(['norm', 'std', 'ged', 'snorm', 'sged', 'sstd', 'jsu', 'sgh', 'sghst'])
    assert all(out) == True
