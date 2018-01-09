
# WrongFoo: WrongFoo

class WrongFoo:
    # self: instance
    # <Dead code detected>

    def __abs__(self):
        return '4'

# x: TypeError
x = (abs(WrongFoo()) + 3)