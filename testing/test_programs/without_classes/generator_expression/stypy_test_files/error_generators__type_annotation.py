
# generators: function
# generators() -> None 

def generators():
    # f: function
    # f(x: int) -> str \/ int 

    def f(x):

        if False:
            return str(x)

        return x

    # x: int; r: list[str \/ int]
    r = [f(x) for x in range(10)]
    # r2: str
    r2 = r[0].capitalize()

generators()