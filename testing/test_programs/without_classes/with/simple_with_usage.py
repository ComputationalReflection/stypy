


class controlled_execution:
    def __enter__(self):
        print "enter the with class"
        return 0

    def __exit__(self, type, value, traceback):
        print "exit the with class"

a = 3

with controlled_execution() as thing:
    a = a + 1
    print thing
