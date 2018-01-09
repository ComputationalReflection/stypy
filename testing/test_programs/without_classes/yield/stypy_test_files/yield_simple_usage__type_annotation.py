
# createGenerator: function
# createGenerator() -> generator[int] 

def createGenerator():
    # mylist: list[int]
    mylist = range(3)
    # i: int
    for i in mylist:
        (yield (i * i))

# x: generator[int]
x = createGenerator()