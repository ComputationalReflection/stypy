class ndenumerate(object):
    def __next__(self):
        return 0

    next = __next__


o = ndenumerate()
r = o.next
r2 = o.__next__
print r
print r2
