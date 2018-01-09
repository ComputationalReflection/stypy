

def function(x):
    def another_function(z):
        return str(z)

    return another_function(x)

ret = function(3)
