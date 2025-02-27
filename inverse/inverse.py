import math
import random

def reciprocal(a):
    if a == 0.0:
        raise ValueError("reciprocal of zero is undefined")
    # 分解a的尾数和指数
    m, e = math.frexp(a)
    # 处理符号
    s = 1.0 if m > 0 else -1.0
    m_abs = abs(m)
    # 初始猜测
    x = 3.0 - 2.0 * m_abs
    # 进行6次牛顿迭代以提升精度
    for _ in range(6):
        x = x * (2.0 - m_abs * x)
    # 调整符号并应用指数
    return s * x * math.ldexp(1.0, -e)

if __name__ == "__main__":
    for i in range(100):
        a = random.random()
        inverse = reciprocal(a)
        print(f"a = {a}, inverse = {inverse}, a * inverse = {a * inverse}")