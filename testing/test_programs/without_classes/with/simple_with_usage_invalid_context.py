


class controlled_execution:
    def __enter__(self):
        print "enter the with class"
        return 0

a = 3

with controlled_execution() as thing:
    a = a + 1
    print thing
