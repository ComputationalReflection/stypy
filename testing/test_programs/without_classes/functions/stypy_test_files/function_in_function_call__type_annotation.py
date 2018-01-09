
# function: function
# function(x: int) -> str 

def function(x):
    # another_function: function
    # another_function(z: int) -> str 

    def another_function(z):
        return str(z)

    # z: int
    return another_function(x)

# x: int; ret: str
ret = function(3)