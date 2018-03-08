
def f():
    return True, 0.0


d = {1: "one",
     2: "two"}

d2 = {(1,1): "pair of ones",
      (2,2): "pair of twos"}

for (k, v) in enumerate(d.values()):
    print str(k) + ", " + str(v)

for (k2, v2) in d.items():
    print str(k2) + ", " + str(v2)

for (k3, v3) in d2.keys():
    print str(k3) + ", " + str(v3)

k4, v4 = (True, 1)
print str(k4) + ", " + str(v4)

k5, v5 = f()
print str(k5) + ", " + str(v5)
