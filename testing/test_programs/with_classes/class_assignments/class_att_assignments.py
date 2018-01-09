

class C:
    def __init__(self):
        pass

    r = "hi"


C.r = 5

c = C()

x = c.r == 5
