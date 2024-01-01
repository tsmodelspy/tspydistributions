# sandwich estimators for standard errors
import numpy as np
import numpy.typing as npt
from typing import Optional, TypeVar

Array = npt.NDArray[np.float64]

def vcov(hessian: Array, scores: Optional[Array] = None, type:str = 'H', adjust:bool = False)->Array:
    allowed_types = ['H', 'OPG', 'QMLE']
    if type not in allowed_types:
        raise ValueError(f"Invalid type. Expected one of: {allowed_types}")
    if type == 'H':
        V = _vcov_h(hessian)
    elif type == 'OPG':
        if scores is None:
            raise ValueError("OPG requires scores")
        V = _vcov_opg(scores, adjust)
    elif type == 'QMLE':
        if scores is None:
            raise ValueError("QMLE requires scores")
        V = _vcov_qmle(scores, hessian, adjust)
    else:
        V = _vcov_h(hessian)
    return V


def _vcov_opg(scores: Array, adjust:bool = False)->Array:
    dims = scores.shape
    k = dims[1]
    n = dims[0]
    #rval = np.matmul(scores.transpose(), scores)
    Q, R = np.linalg.qr(scores)
    rval = np.linalg.inv(np.matmul(R.transpose(), R))
    if adjust:
        rval = n/(n - k) * rval
    return rval

def _meat(scores: Array, adjust:bool = False)->Array:
    dims = scores.shape
    k = dims[1]
    n = dims[0]
    rval = scores.transpose() @ scores
    if adjust:
        rval = n/(n - k) * rval
    return rval


def _vcov_qmle(scores: Array, hessian: Array, adjust:bool = False)->Array:
    bread = np.linalg.inv(hessian)
    meat = _meat(scores, adjust)
    V = bread @ meat @bread
    return V

def _vcov_h(hessian: Array)->Array:
    V = np.linalg.inv(hessian)
    return V
