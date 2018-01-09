
# function: function
# function(a: int) -> str \/ int \/ bool 

def function(a):

    if (a > 0):
        return 'Positive'


    if (a < 0):
        return a


    if (a == 0):
        return False


# a: int; x: str \/ int \/ bool
x = function(3)