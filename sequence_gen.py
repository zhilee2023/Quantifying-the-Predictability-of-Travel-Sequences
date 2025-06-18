import numpy as np
import math
import os
import json
import matplotlib.pyplot as plt
from ctw_estimate import CTWEntropy
from collections import defaultdict
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def generate_gaussian_markov(coeffs, length, sigma_z=1.0):
    """
    生成一个 AR(p) 过程的 Gaussian Markov 序列，改进了初始条件和数值稳定性。
    """
    p = len(coeffs)
    X = np.zeros(length)

    # 使用随机初始条件而非全零，提高数值稳定性
    if p > 0:
        X[:p] = np.random.normal(0.0, sigma_z**2 / (1 + np.sum(coeffs)**2), p)

    Z = np.random.normal(loc=0.0, scale=sigma_z, size=length)

    for t in range(p, length):
        # 添加数值限制防止溢出
        prev_values = X[t-p:t][::-1]
        X[t] = np.dot(coeffs, prev_values) + Z[t]
        # 检查并限制X[t]的范围
        if np.abs(X[t]) > 1e30:
            X[t] = np.sign(X[t]) * 1e30  # 防止inf转换为nan
    return X

def safe_log2(x, eps=1e-20):
    return np.log2(np.maximum(x, eps))



def generate_random_stable_ar_coeffs(p, max_radius=0.98, epsilon=1e-12, max_trials=1000000):
    """
    生成稳定 AR(p) 系数，设置最大尝试次数避免长时间阻塞。
    """
    for trial in range(1, max_trials+1):
        # 随机采样根
        angles = np.random.uniform(0, 2*np.pi, size=p)
        radii  = np.random.uniform(0, max_radius, size=p)
        roots  = radii * np.exp(1j * angles)

        # 初步过滤：所有根都要 |root| < 1 - epsilon
        if np.any(np.abs(roots) >= 1 - epsilon):
            continue

        # 构造多项式，提取 AR 系数
        poly_coeffs = np.poly(roots)      # [1, c1, …, cp]
        ar_coeffs   = -poly_coeffs[1:]    # ai = -ci
        ar_coeffs   = np.real(ar_coeffs)

        # 数值验证：再次检查所有根都在单位圆内
        eig = np.roots(poly_coeffs)
        if np.max(np.abs(eig)) < 1 - epsilon:
            return ar_coeffs

    raise RuntimeError(
        f"在 {max_trials} 次尝试后仍未找到稳定系数，"
        "请检查 p、max_radius、epsilon 的设置是否合理。"
    )


def S_X(omega, coeffs, sigma_z):
    """
    改进的功率谱密度计算，增加分母保护。
    """
    p = len(coeffs)
    exp_terms = np.array([np.exp(-1j * omega * i) for i in range(1, p+1)])
    denominator = 1 - np.dot(coeffs, exp_terms)
    denom_mag_sq = np.abs(denominator) ** 2 + 1e-20  # 防止除零
    return (sigma_z ** 2) / denom_mag_sq

def compute_rate_distortion(coeffs, sigma_z, theta_vals, n_freq=20000):
    """
    利用水填充法计算 AR(p) 过程的 Rate-Distortion 数值对
    参数:
      coeffs: AR(p) 系数列表
      sigma_z: 白噪声标准差
      theta_vals: 水位 theta 取值数组
      n_freq: 积分时频率采样点数
    返回:
      D_vals: 对应每个 theta 的平均失真 D 数组
      R_vals: 对应每个 theta 的编码率 R 数组（单位 bits）
    """
    omega = np.linspace(-np.pi, np.pi, n_freq)
    S_vals = S_X(omega, coeffs, sigma_z)

    D_vals = []
    R_vals = []
    epsilon = 1e-12  # 防止对0取log
    for theta in theta_vals:
        D = (1 / (2 * np.pi)) * np.trapz(np.minimum(S_vals, theta), omega)
        D_vals.append(D)
        integrand = np.maximum(0, np.log2((S_vals + epsilon) / theta))
        R = (1 / (4 * np.pi)) * np.trapz(integrand, omega)
        R_vals.append(R)
    return np.array(D_vals), np.array(R_vals)

def sliding_window_batches_1d(x, T, stride=1):
    """
    生成一维序列的滑动窗口 batch

    参数:
    x: 输入一维序列，长度为 N
    T: 每个子序列的长度
    stride: 滑动步长，默认为1

    返回:
    一个二维数组，形状为 (num_windows, T)，其中 num_windows = (N - T) // stride + 1
    """
    N = len(x)
    num_windows = (N - T) // stride + 1
    shape = (num_windows, T)
    strides = (x.strides[0] * stride, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def sliding_window_batches(X, T, stride=1):
    """
    对二维数组 X 的每一列应用滑动窗口。

    参数:
    X: 输入二维数组，形状为 (len, feature_len)
    T: 每个子序列的长度
    stride: 滑动步长，默认为1

    返回:
    一个三维数组，形状为 (num_windows, T, feature_len)
    """
    len_x, feature_len = X.shape
    num_windows = (len_x - T) // stride + 1
    result = np.zeros((num_windows, T, feature_len))
    for i in range(feature_len):
        result[:, :, i] = sliding_window_batches_1d(X[:, i], T, stride)
    return result


def cal_entropy_rate(data,D_vals,R_vals,dist):
    # Convert the sequence to int, and reindex
    K=max(data)+1
    H=CTWEntropy(max_symbol=K).calculate_entropy_rate(data)
    pi=find_p(H, K)
    d_base = create_baseline_spline(D_vals,R_vals)
    R_=d_base(dist)
    Pi=find_p(R_, K)
    return H,R_,pi,Pi

def f_e(p, Z):
    """
    Computes f_e(p) = H_b(p) + p * log2(|Z|-1)

    where H_b(p) = -p*log2(p) - (1-p)*log2(1-p)
    """
    # Ensure valid domain for entropy
    if p == 0 or p == 1:
        H_b = 0.0
    else:
        H_b = -p * safe_log2(p) - (1 - p) * safe_log2(1 - p)
    return H_b + p * safe_log2(Z - 1)


def find_p(target_fe, Z, tol=1e-10):
    """
    Given target f_e value and alphabet size Z (integer > 1),
    find p in (0, 1) such that f_e(p, Z) = target_fe.
    Uses Brent's method for root finding.
    """
    # Define the function whose root we want


    # Check feasibility
    fe_min = f_e(0.0, Z)
    fe_max = f_e(1.0, Z)
    # if not (fe_min <= target_fe <= fe_max):
    #     raise ValueError(f"target_fe must be between {fe_min:.6f} and {fe_max:.6f}")
    target_fe=float(np.clip(target_fe, fe_min, fe_max))
    def func(p):
        return f_e(p, Z) - target_fe
    # Find root in (0, 1)
    p_root = brentq(func, 1e-20, 1 - 1e-20, xtol=tol)
    return 1-p_root



def save_parameters_to_json(params, output_dir, timestamp):
    """将参数保存到 JSON 文件中。"""
    filename = os.path.join(output_dir, f"parameters_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    #print(f"Parameters saved to: {filename}")

def generate_gaussian_markov_sequence(N, D, R, noise_std=1.0):
    """
    生成高斯马尔可夫序列并计算 SigmaZ 矩阵。

    参数:
      N         : 序列长度
      D         : 序列维度
      R         : AR 阶数
      noise_std : 噪声的标准差

    返回:
      sequence  : 生成的高斯马尔可夫序列，形状为 (N, D)
      SigmaZ    : 序列的协方差矩阵，形状为 (D, D)
    """
    def generate_stable_ar_matrices(p, N, max_radius=0.3, max_iter=1000000):
        """
        生成稳定的AR(p)系数矩阵列表
        参数:
        p: AR阶数
        N: 向量维度
        max_radius: 系数矩阵元素范围控制
        """
        for _ in range(max_iter):
            # 生成随机系数矩阵（限制幅值以增强稳定性）
            A_matrices = [np.random.uniform(-max_radius/p, max_radius/p, (N,N)) for _ in range(p)]

            # 构建块伴随矩阵并验证特征值
            companion = np.zeros((p*N, p*N), dtype=np.complex128)
            companion[:N, :] = np.hstack(A_matrices)
            for i in range(1, p):
                companion[i*N:(i+1)*N, (i-1)*N:i*N] = np.eye(N)

            eig_vals = np.linalg.eigvals(companion)
            if np.max(np.abs(eig_vals)) < 1 - 1e-10:
                return A_matrices
        raise ValueError("无法生成稳定AR矩阵，请调整max_radius或max_iter")


    # 1. 生成稳定的AR系数矩阵
    A_matrices = generate_stable_ar_matrices(R, D)

    #p, N = len(A_matrices), A_matrices[0].shape[0]
    X = np.zeros((N, D))

    # 初始化前p个时间步
    for t in range(R):
        X[t] = np.random.normal(0, noise_std, D)

    # AR过程迭代
    for t in range(R, N):
        for i in range(R):
            X[t] += A_matrices[i] @ X[t-i-1]
        X[t] += np.random.normal(0, noise_std, D)
    # 3. 计算 SigmaZ 矩阵
    #SigmaZ = np.cov(sequence.T)
    return X,A_matrices




# ========== 2. 计算率失真曲线 ==========
def compute_rate_distortion_vector(A_matrices, sigma_z, theta_vals=np.logspace(-3, 2, 50), n_freq=1000):
    """
    计算多维AR过程的RD曲线
    参数:
      theta_vals: 水填充参数（指数间隔优化[4](@ref)）
      n_freq: 频率采样点数
    """
    p, N = len(A_matrices), A_matrices[0].shape[0]
    omega = np.linspace(-np.pi, np.pi, n_freq)
    eigenvals = np.zeros((n_freq, N))
    I = np.eye(N, dtype=np.complex128)

    # 计算功率谱特征值
    for idx, w in enumerate(omega):
        H = I.copy()
        for i in range(p):
            H -= A_matrices[i] * np.exp(-1j * w * (i+1))
        H_inv = np.linalg.pinv(H)  # 伪逆增强稳定性[3](@ref)
        S = (sigma_z**2) * H_inv @ H_inv.conj().T
        S = 0.5 * (S + S.conj().T)  # 强制埃尔米特性[2](@ref)
        eigenvals[idx] = np.clip(np.linalg.eigvalsh(S), 0, None)  # 确保非负

    # 水填充算法
    D_vals, R_vals = [], []
    for theta in theta_vals:
        ratio = eigenvals / (theta + 1e-15)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.log2(np.where(ratio > 0, ratio, 1))
        R_integrand = 0.5 * np.sum(np.maximum(0, log_ratio), axis=1)
        D_integrand = np.sum(np.minimum(eigenvals, theta), axis=1)

        R = np.trapz(R_integrand, omega) / (2*np.pi)
        D = np.trapz(D_integrand, omega) / (2*np.pi)
        R_vals.append(R)
        D_vals.append(D)
    return np.array(D_vals), np.array(R_vals)



def compute_distances(R_vals, D_vals, sample_R, sample_D, return_per_sample=False):
    """
    计算样本点到 baseline 曲线的均方误差 (MSE)

    参数：
      R_vals, D_vals       baseline 曲线原始点集合
      sample_R, sample_D   样本点集合
      return_per_sample    是否同时返回每个点的平方误差

    返回：
      如果 return_per_sample=False，返回一个标量：所有点的 MSE；
      如果 return_per_sample=True，返回 (per_sample_se, mse)。
    """
    # 1. 创建 baseline 样条函数
    d_base = create_baseline_spline(R_vals, D_vals)

    # 2. 计算插值点的 D 值
    sample_R = np.array(sample_R, dtype=float)
    baseline_at_samples = d_base(sample_R)

    # 3. 计算平方误差
    per_sample_se = (np.array(sample_D, dtype=float) - baseline_at_samples) ** 2
    mse = per_sample_se.mean() if per_sample_se.size else 0.0

    if return_per_sample:
        return per_sample_se, mse
    else:
        return mse



def create_baseline_spline(R_vals, D_vals):
    """
    根据原始 (R_vals, D_vals) 点集，去重取平均并按 R 排序，
    构造并返回一个 scipy.interpolate.CubicSpline 对象。

    参数：
      R_vals, D_vals: 一维可迭代，原始基线曲线的 R 和 D 值

    返回：
      d_base: CubicSpline 对象，可用于 d_base(r) 获得插值后的 D 值
    """
    R = np.array(R_vals, dtype=float)
    D = np.array(D_vals, dtype=float)

    # 去重并平均
    vals = defaultdict(list)
    for r, d in zip(R, D):
        vals[r].append(d)
    unique_R = np.array(list(vals.keys()), dtype=float)
    unique_D = np.array([np.mean(v) for v in vals.values()], dtype=float)

    # 排序
    idx = np.argsort(unique_R)
    unique_R, unique_D = unique_R[idx], unique_D[idx]

    # 如果不足两点，则构造常数函数
    if unique_R.size < 2:
        const_val = unique_D[0] if unique_R.size == 1 else 0.0
        return lambda x: np.full_like(x, fill_value=const_val, dtype=float)

    # 构造三次样条
    return CubicSpline(unique_R, unique_D)