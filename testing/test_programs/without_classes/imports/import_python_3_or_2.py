
import sys

if sys.version_info[0] >= 3:
    import pickle
    basestring = str
    import builtins
else:
    import cPickle as pickle
    import __builtin__ as builtins

