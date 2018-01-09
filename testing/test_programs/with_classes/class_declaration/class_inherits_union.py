
class A:
    a_Att = 3

    def ma(self):
        return "A"

class B:
    def mb(self):
        return "B"

if True:
    base = A
else:
    base = B


class Simple(base):
    sample_att = 3
    (a,b) = (6,7)

    def from_a(self):
        return self.a_Att

    def sample_method(self):
        self.att = "sample"
        return self.att


x = Simple()
y = x.sample_method()
z = x.ma()
w = x.mb()
k = x.from_a()



