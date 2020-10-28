# encoding: utf-8


class CachedFunc:
    """
    将某个函数进行封装得到新的函数。
    新的函数会缓存结果，直到调用reset方法为止。
    在此期间，直接返回缓存的结果，而不调用原函数，即使参数改变。
    """
    EMPTY = object()

    def __init__(self, func):
        self.func = func
        self._result = CachedFunc.EMPTY

    def __call__(self, *args, **kwargs):
        if self._result is CachedFunc.EMPTY:
            self._result =  self.func(*args, **kwargs)
        return self._result

    def reset(self):
        """清空缓存的结果"""
        self._result = CachedFunc.EMPTY


if __name__ == '__main__':
    pass
