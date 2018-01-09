

class C:
    def __init__(self):
        pass

    def method(self):
        self.r = "str"

c = C()

c.method()

x = c.r == 5


class Counter:
    count = 0

    def __init__(self):
        pass

    def inc(self, value):
        self.count += value
        return self.count

obj = Counter()
sum = obj.inc(1) + obj.inc(0.2)

if obj:
    resul = obj.inc(1)
else:
    resul = obj.inc(0.5)


