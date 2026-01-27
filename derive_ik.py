import sympy as sp
import math

def dh_transform_matrix(alpha, a, d, theta):
    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# 定义符号变量
q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5')
offset3, offset4 = sp.symbols('offset3 offset4')  # 关节偏移量
nx, ny, nz, ox, oy, oz, ax, ay, az, px, py, pz = sp.symbols('nx ny nz ox oy oz ax ay az px py pz')
T_final = sp.Matrix([
        [nx, ox, ax, px],
        [ny, oy, ay, py],
        [nz, oz, az, pz],
        [0, 0, 0, 1]
    ])

# 定义 DH 参数符号
alpha0, alpha1, alpha2, alpha3, alpha4 = sp.symbols('alpha0 alpha1 alpha2 alpha3 alpha4')
a0, a1, a2, a3, a4 = sp.symbols('a0 a1 a2 a3 a4')
d0, d1, d2, d3, d4, d5 = sp.symbols('d0 d1 d2 d3 d4 d5')

# 构建变换矩阵链 (5自由度)
T0 = dh_transform_matrix(0, a0, d0, 0)  # 基座到关节1的固定变换
T1 = dh_transform_matrix(math.pi/2, a1, d1, q1)
T1_simplified = sp.simplify(T1)
T1_inv = T1.inv()
T1_inv_simplified = sp.simplify(T1_inv)
T2 = dh_transform_matrix(0, a2, d2, q2)
T3 = dh_transform_matrix(0, a3, d3, q3)
T4 = dh_transform_matrix(math.pi/2, a4, d4, q4)
T5 = dh_transform_matrix(0, 0, d5, q5)  # 关节5到末端

print("\n" + "="*50)
print("T1 的逆矩阵 (T1^{-1}):")
print("="*50)
sp.pprint(T1_simplified, use_unicode=True)

print("\n" + "="*50)
print("T1 的逆矩阵 * T_final:")
print("="*50)
sp.pprint(sp.simplify(T1_inv_simplified * T_final), use_unicode=True)

# 计算总变换矩阵
T_total = T0 * T1 * T2 * T3 * T4 * T5
T_inv = T1_inv_simplified * T_total
T_inv_simplified = sp.simplify(T_inv)

# 提取姿态矩阵和位置向量
R = T_inv_simplified[:3, :3]  # 旋转矩阵
P = T_inv_simplified[:3, 3]   # 位置向量

# 分解旋转矩阵为 n, o, a 向量
n_vector = sp.Matrix([R[0, 0], R[1, 0], R[2, 0]])
o_vector = sp.Matrix([R[0, 1], R[1, 1], R[2, 1]])
a_vector = sp.Matrix([R[0, 2], R[1, 2], R[2, 2]])

# 简化表达式
n_vector = sp.simplify(n_vector)
o_vector = sp.simplify(o_vector)
a_vector = sp.simplify(a_vector)
P = sp.simplify(P)

# 打印结果
# print("末端执行器姿态和位置矩阵:")
# print("T_total =")
# sp.pprint(T_total, use_unicode=True)

# print("\n姿态向量 n =")
# sp.pprint(n_vector, use_unicode=True)

# print("\n姿态向量 o =")
# sp.pprint(o_vector, use_unicode=True)

# print("\n姿态向量 a =")
# sp.pprint(a_vector, use_unicode=True)

# print("\n位置向量 p =")
# sp.pprint(P, use_unicode=True)

# 提取各个分量
print("\n各分量表达式:")
print(f"nx = {n_vector[0]}")
print(f"ny = {n_vector[1]}")
print(f"nz = {n_vector[2]}")
print(f"ox = {o_vector[0]}")
print(f"oy = {o_vector[1]}")
print(f"oz = {o_vector[2]}")
print(f"ax = {a_vector[0]}")
print(f"ay = {a_vector[1]}")
print(f"az = {a_vector[2]}")
# print(f"px = {P[0]}")
# print(f"py = {P[1]}")
# print(f"pz = {P[2]}")