
import os
import sys

import types

def test_package(package, depth):
    if package is None:
        f = sys._getframe(1 + depth)
        package_path = f.f_locals.get('__file__', None)
        if package_path is None:
            raise AssertionError
        package_path = os.path.dirname(package_path)
        package_name = f.f_locals.get('__name__', None)
    elif isinstance(package, type(os)):
        package_path = os.path.dirname(package.__file__)
        package_name = getattr(package, '__name__', None)
    else:
        package_path = str(package)

test_package("os", 1)
test_package(os, 1)
test_package(None, 1)

r = "3"

if not r is None:
    r2 = 3
else:
    r2 = 3.0

rb = None

if not rb is None:
    r3 = 3
else:
    r3 = 3.0



