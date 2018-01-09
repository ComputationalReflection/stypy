class WrongFoo:
    def __abs__(self):
        return "4"


x = (abs(WrongFoo()) + 3)  # Not reported, runtime crash
