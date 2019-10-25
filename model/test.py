import time
from functools import wraps

def super_func(c,d):
    '''
    Decorator that reports the execution time.
    '''
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)+c-d
        return wrapper
    return decorate


@super_func(c=1,d=2)
def add(x, y):
    return x + y

print(add(1,3))
