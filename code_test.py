

class Vector3f(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

mergeable_types = [int, float, long, str, complex]


def only_mergeable_types(obj):
    vals = obj.__dict__.values()

    for v in vals:
        if type(v) not in mergeable_types:
            return False

    return True

v = Vector3f()

print only_mergeable_types(v)