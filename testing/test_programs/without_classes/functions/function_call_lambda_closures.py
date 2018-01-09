


d = dict()

d[1] = "one"
d[2] = "two"

cast = dict()

for key in d.keys():
    cast[key] = lambda x, k=key: str(k) + str(x)
    r = cast[key](10)
    print r

for key, val in d.items():
    if val not in cast:
         cast[val] = key

print d
print cast