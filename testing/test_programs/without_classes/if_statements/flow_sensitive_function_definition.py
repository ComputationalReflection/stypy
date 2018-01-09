#import random
a = 3
#condition = random.randint(0, 1) == 0
condition = a > 0
if condition:
    f = lambda x: x
else:
    f = lambda x,y: x+y
f(1)
f()