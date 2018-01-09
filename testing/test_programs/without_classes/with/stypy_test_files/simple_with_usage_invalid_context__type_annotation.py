
# controlled_execution: controlled_execution

class controlled_execution:
    # self: instance
    # <Dead code detected>

    def __enter__(self):
        print 'enter the with class'
        return 0

# a: int
a = 3
with controlled_execution() as thing:
    a = (a + 1)
    print thing