from time import time
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree, NearestNeighbors
from vincenty_cuda_nns import CudaTree
from brute_cuda import brute_cuda
from brute_cpu import brute_cpu, compiled_vincenty


def test_tree(K=2, LS=3):
    df = gpd.read_file('datasets/UK.geojson')
    X = np.stack(df['geometry']).astype(np.float32)

    print('{0} neighbors of {1} points'.format(K, len(X)))
    print()
    print('===== Run =====')

    t0 = time()
    bt1 = BallTree(X, leaf_size=LS, metric=compiled_vincenty)
    t1 = time()
    print('SkLearn build: {0:.3g} sec'.format(t1 - t0))

    t1 = time()
    sk_dist, sk_ind = bt1.query(X, K)
    t2 = time()
    print('SkLearn query: {0:.3g} sec'.format(t2 - t1))
    print()

    t2 = time()
    cuda_tree = CudaTree(X, leaf_size=LS)
    t3 = time()
    print('CUDA build  : {0:.3g} sec'.format(t3 - t2))

    t3 = time()
    cuda_dist, cuda_ind = cuda_tree.query(n_neighbors=K)
    t4 = time()
    print('CUDA query  : {0:.3g} sec'.format(t4 - t3))
    print()

    t4 = time()
    brute_dist = brute_cuda(X)
    t5 = time()
    print('Brute CUDA : {0:.3g} sec'.format(t5 - t4))
    print()

    t5 = time()
    cpu_dist = brute_cpu(X)
    t6 = time()
    print('Brute parallel CPU: {0:.3g} sec'.format(t6 - t5))
    print()

    t6 = time()
    nn = NearestNeighbors(
        algorithm='ball_tree',
        metric=compiled_vincenty,
        n_jobs=-1,
        leaf_size=3
    ).fit(X)
    t7 = time()
    print('SkLearn parallel build: {0:.3g} sec'.format(t7 - t6))

    t7 = time()
    res = nn.kneighbors_graph(X, n_neighbors=K, mode='distance')
    sk_cpu_dist = res.data.reshape((res.shape[0], K))
    sk_cpu_ind = res.indices.reshape((res.shape[0], K))
    t8 = time()
    print('SkLearn parallel query: {0:.3g} sec'.format(t8 - t7))
    print()

    print('===== Comparing =====')
    print('Brute CUDA distance == SkLearn distance:',
          np.allclose(sk_dist[:, 1], brute_dist))
    print('Brute CUDA distance == CUDA distance:',
          np.allclose(cuda_dist[:, 1], brute_dist))
    print('Brute CUDA distance == Brute parallel CPU distance:',
          np.allclose(cpu_dist, brute_dist))
    print('Brute CUDA distance == SkLearn parallel CPU distance:',
          np.allclose(sk_cpu_dist[:, 1], brute_dist))
    print()

    print('CUDA indexes = SkLearn indexes:',
          np.allclose(sk_ind, cuda_ind, rtol=0))
    print('CUDA indexes = SkLearn parallel CPU indexes:',
          np.allclose(sk_cpu_ind, cuda_ind, rtol=0))
    print()

    print('===== Timing =====')
    print('Brute CUDA time: {0:.3g} sec'.format(t5 - t4))
    print('CUDA time: {0:.3g} sec'.format(t4 - t2))
    print('SkLearn time: {0:.3g} sec'.format(t2 - t0))
    print('Brute parallel CPU time: {0:.3g} sec'.format(t6 - t5))
    print('SkLearn parallel CPU time: {0:.3g} sec'.format(t8 - t6))


if __name__ == '__main__':
    test_tree()
