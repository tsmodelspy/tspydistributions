"""
Demo: GHST Distribution
-----------------------
"""
from plotnine import ggplot, aes, geom_area
from tspydistributions import pdqr

def ghst_left(x):
    return pdqr.dsghst(x, mu=2, sigma=1, skew=-20, shape=10)[0]

def ghst_right(x):
    return pdqr.dsghst(x, mu=-2, sigma=1, skew=20, shape=10)[0]

plot = (ggplot(None,aes([-6,6])) + 
        geom_area(stat = "function", fun = ghst_left, fill = "cadetblue", alpha = 0.4, xlim = [-6, 6]) + 
        geom_area(stat = "function", fun = ghst_right, fill = "darkgrey", alpha = 0.4, xlim = [-6, 6]))
print(plot)