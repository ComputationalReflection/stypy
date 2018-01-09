from math import cos as aliased

aliased = []  # Wrong error


def alias():
    r = aliased(0.5)  # Not detected


alias()
