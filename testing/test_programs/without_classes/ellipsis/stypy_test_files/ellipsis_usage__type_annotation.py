
# Foo: Foo

class Foo:
    # self: instance
    # __getitem__(*args: tuple[int]) -> tuple[int] /\ __getitem__(*args: tuple[slice[]]) -> tuple[slice[]] /\ __getitem__(*args: tuple[tuple]) -> tuple[tuple] 

    def __getitem__(self, *args):
        # self: instance
        print args
        return args

# x: Foo instance
x = Foo()
# args: tuple[int]; r1: tuple[int]
r1 = x[1]
# args: tuple[slice[]]; r2: tuple[slice[]]
r2 = x[1:]
# args: tuple[tuple]; r3: tuple[tuple]
r3 = x[1:, :]
# r4: tuple[tuple]; args: tuple[tuple]
r4 = x[1:, 20:10:(-2), ...]