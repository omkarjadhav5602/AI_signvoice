"""Small Python example showing how the `global` keyword works."""

a = 10


def g_var():
    global a
    a = 15
    print("inside function", a)


g_var()
print("outside", a)
