import cupy

def median(x, axis):
    """配列(x)の軸(axis)に沿った中央値"""
    xp = cupy.get_array_module(x)
    n = x.shape[axis]
    s = xp.sort(x, axis)
    m_odd = xp.take(s, n // 2, axis)
    if n % 2 == 1:  # 奇数個
        return m_odd
    else:  # 偶数個のときは中間の値
        m_even = xp.take(s, n // 2 - 1, axis)
        return (m_odd + m_even) / 2

# 動作例
cx = cupy.array([[2, 1, 2, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
mc0 = median(cx, 0)  # [2, 2, 2, 4]
mc1 = median(cx, 1)  # [2, 2.5, 2.5]

# cupyからnumpyの配列に変換
nx = cupy.asnumpy(cx)
mn0 = median(nx, 0)  # cupy 配列と同じように使える
# numpyからcupyの配列に変換
cx = cupy.asarray(nx)


import numpy
from cupy.testing import assert_allclose
assert_allclose(mc0, numpy.median(nx, 0))
assert_allclose(mc1, numpy.median(nx, 1))
assert_allclose(mc0, numpy.median(nx, 0))
assert_allclose(mc1, numpy.median(nx, 1))
