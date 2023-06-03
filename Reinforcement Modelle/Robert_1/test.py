from functions import *


class myClass():
    def __init__(self, function):
        self.output = function()
        self.c = 3


test = myClass(function=func_b)
print(test.output)
