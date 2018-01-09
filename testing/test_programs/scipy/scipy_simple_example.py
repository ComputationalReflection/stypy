

from scipy.special import jv
from scipy.optimize import minimize
from numpy import linspace

f = lambda x: -jv(3, x)
sol = minimize(f, 1.0)
x = linspace(0, 10, 5000)

