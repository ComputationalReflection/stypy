
# random: module
import random

# Counter: Counter

class Counter:
    # self: instance
    count = 0
    # inc(value: int) -> int /\ inc(value: float) -> float 

    def inc(self, value):
        # count: int \/ float; self: instance
        self.count += value
        return self.count

# bitwise_or: function
# bitwise_or(counter: Counter instance \/ Counter instance, n: int) -> int 

def bitwise_or(counter, n):
    # x: int \/ float
    x = counter.count
    return (counter.count | n)

# flow_sensitive: function
# flow_sensitive(obj: Counter instance \/ Counter instance, condition: bool) -> int 

def flow_sensitive(obj, condition):

    if condition:
        # value: int
        obj.inc(1)
    else:
        # value: float
        obj.inc(0.5)

    # counter: Counter instance \/ Counter instance; n: int
    return bitwise_or(obj, 3)

# obj: Counter instance
obj = Counter()
# obj: Counter instance; condition: bool
flow_sensitive(obj, (random.randint(0, 1) == 0))