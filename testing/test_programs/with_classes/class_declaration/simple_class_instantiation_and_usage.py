

class Simple:
    sample_att = 3
    (a,b) = (6,7)

    def __init__(self):
        pass

    def sample_method(self):
        self.att = "sample"


s = Simple()

s.sample_method()

result = s.att
result2 = s.b

