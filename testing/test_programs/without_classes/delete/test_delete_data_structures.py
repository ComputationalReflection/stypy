# coding=utf-8

a = 3

t = ("z","a")

l = range(5)

d = {
    'a': 1,
    'b': 2,
}

# Right
del l[1]
del d['b']

if True:
    del d['a']
else:
    del l[2]

# Wrong
del t[1]
del a['b']

if True:
    del t['a']
else:
    del a[2]