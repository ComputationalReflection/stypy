
# Simple: Simple

class Simple:
    sample_att = 3
    (a, b) = (6, 7)
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

    # sample_method() -> None 

    def sample_method(self):
        # self: instance; att: str
        self.att = 'sample'

# s: Simple instance
s = Simple()
s.sample_method()
# result: str
result = s.att
# result2: int
result2 = s.b