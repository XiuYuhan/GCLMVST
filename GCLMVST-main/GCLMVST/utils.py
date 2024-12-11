import os
import ot
import random

import scipy
import torch
import numpy as np
import numba


def fix_seed(seed=2023):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_outliers(data):
    z_scores = np.abs(scipy.stats.zscore(data))
    outlier_indices = np.where(z_scores > 3)
    return outlier_indices



def mclust(adata, arg, refine=False):
    print("Outliers indices:", adata.obsm['GCLMVST'])
    aa=adata.obsm['GCLMVST'].copy()
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=40, kernel='linear', eigen_solver='auto')  # 0.60,450
    embedding = kpca.fit_transform(aa)
    adata.obsm['emb_pca'] = embedding
    import rpy2.robjects as r_objects
    r_objects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = r_objects.r['set.seed']
    r_random_seed(arg.seed)
    rmclust = r_objects.r['Mclust']

    cov_matrix = np.cov(adata.obsm['emb_pca'], rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print("Eigenvalues:\n", eigenvalues)

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), arg.n_domain, 'EEE')

    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    if refine:
        new_type = refine_label(adata, radius=arg.n_refine, key='mclust')
        adata.obs['mclust'] = new_type
    return adata



def refine_label(adata, radius=0, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


# 使用 Numba 的装饰器，用于加速 Python 代码，特别是对数值计算密集型任务的优化
# 计算输入数据矩阵 X 中所有点（数据样本）之间的欧几里得距离，并返回一个邻接矩阵（距离矩阵）
@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    # n = 输入数据 X 中的数据点数量，即 X.shape[0]
    n = X.shape[0]
    # 初始化一个形状为 (n, n) 的空的二维数组，类型为 float32。这个数组将用来存储每对数据点之间的距离
    # adj 将是对称的，因为距离矩阵是对称的，即 adj[i][j] 等于 adj[j][i]
    adj = np.empty((n, n), dtype=np.float32)
    # 遍历每一个数据点 i
    for i in numba.prange(n):
        # 对于每个数据点 i，再次遍历所有数据点 j，计算 i 和 j 之间的距离。
        for j in numba.prange(n):
            # 调用 euclid_dist(X[i], X[j]) 计算数据点 i 和 j 之间的欧几里得距离，并将结果存储在 adj[i][j] 中
            adj[i][j] = euclid_dist(X[i], X[j])
    # 邻接矩阵
    return adj


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
              max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    return c3


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                  max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        # 转换为 NumPy 数组
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        # print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        # 根据颜色通道的方差对颜色进行加权平均，得到新的灰度值 c3
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        # 对 c3 进行标准化处理
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        # 计算 z 轴的缩放因子
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        # 计算最终的 z 值
        z = c4 * z_scale
        # 将 z 转换为列表形式
        z = z.tolist()
        # print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        # 将 x, y, z 合并为二维数组 X 并转换为 float32 类型
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        # 如果不使用组织学信息，只使用 x 和 y 坐标
        print("Calculateing adj matrix using xy only...")
        # 将 x 和 y 合并为二维数组 X 并转换为 float32 类型
        X = np.array([x, y]).T.astype(np.float32)
    # 计算 X 中点之间的距离，并返回邻接矩阵
    return pairwise_distance(X)


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        print("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        print("recommended l = ", str(start))
        return start
    elif np.abs(p_high - p) <= tol:
        print("recommended l = ", str(end))
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        # print("Run " + str(run) + ": l [" + str(start) + ", " + str(end) + "], p [" + str(p_low) + ", " + str(
        #     p_high) + "]")
        if run > max_run:
            # print("Exact l not found, closest values are:\n" + "l=" + str(start) + ": " + "p=" + str(
            #     p_low) + "\nl=" + str(end) + ": " + "p=" + str(p_high))
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            # print("recommended l = ", str(mid))
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return np.mean(np.sum(adj_exp, 1)) - 1