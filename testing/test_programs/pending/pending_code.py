# def function(x):
#     return x[0]
#
# def f2(x):
#     function(x)
#
# y = f2(3)

#
# x = function(function(3)) /2
# #x = function(True)

# class Foo:
#     z = 6
#     def __init__(self):
#         pass
#
#     def anyadeAlgo(self):
#         #self.a = "3"
#         Foo.a = "3"
#
# f = Foo()
# g = Foo()
# #f.a = 3
# #f.anyadeAlgo()
# Foo.a = "3"
# x = f.a
#
# if f.a:
#     g.a = 0
# if x:
#     x2 = g.a
#
# t = Foo.a
#
# def ffff():
#     pass
#
# Foo.nuevo = ffff
#
# zz = f.nuevo
#
# l = range(6)
#
# other_l = map(lambda x: str(x), l)
#
# factorial = lambda n: 1 if n == 1 else factorial(n-1)*n
#
# fact = factorial(2)


# # Recursion
# def factorial(n):
#     # Notice, the SSA algorithm does no allow return in if/else, while, for... bodies.
#     # An assignment must be performed, and return must be placed outside the control flow statement.
#     # Thus, an AST transformation is required.
#     if n == 0 or n == 1:
#         returned_value = 1
#     else:
#         returned_value = n * factorial(n - 1)
#     return returned_value
#
#
# fact_int = factorial(1000)  # int
# fact_float = factorial(1000.0)  # int \/ float

a = 4

def f():

    print a

    global a

    a = 5
    print a

f()
