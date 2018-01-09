
# controlled_execution: controlled_execution

class controlled_execution:
    # self: instance
    # __enter__() -> int 

    def __enter__(self):
        # self: instance
        print 'enter the with class'
        return 0

    # __exit__(type: NoneType, value: NoneType, traceback: NoneType) -> None 

    def __exit__(self, type, value, traceback):
        # self: instance
        print 'exit the with class'

# a: int
a = 3
# traceback: NoneType; type: NoneType; value: NoneType
with controlled_execution():
    # a: int
    a = (a + 1)