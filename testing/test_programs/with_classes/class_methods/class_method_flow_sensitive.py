

import random

class Counter:
    count = 0
    def inc(self, value):
        self.count += value
        return self.count


def bitwise_or(counter, n):
    x = counter.count
    return counter.count | n

def flow_sensitive(obj, condition):
    if condition:
        obj.inc(1)
    else:
        obj.inc(0.5)
    return bitwise_or(obj, 3)

obj = Counter()
flow_sensitive(obj, random.randint(0, 1) == 0)
