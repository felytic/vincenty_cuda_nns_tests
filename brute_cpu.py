from numba import njit, prange, guvectorize
from cuda_friendly_vincenty import vincenty
import numpy as np

wrap = njit(parallel=True)
wrapped_vincenty = wrap(vincenty)


@guvectorize('void(int64, float32[:,:], float32[:])', '(),(n, m) -> ()',
             target='parallel', cache=True)
def get_min_distances(idx, points, result):
    l = len(points)
    min_dist = -1

    for i in prange(l):
        if i != idx:
            distance = wrapped_vincenty(points[idx, 0], points[idx, 1],
                                        points[i, 0], points[i, 1])

            if (min_dist == -1) or (distance < min_dist):
                min_dist = distance

    result[0] = min_dist


@njit
def compiled_vincenty(point1, point2):
    return wrapped_vincenty(point1[0], point1[1], point2[0], point2[1])


def plain_vincenty(point1, point2):
    return vincenty(point1[0], point1[1], point2[0], point2[1])


def brute_cpu(array):
    return get_min_distances(range(len(array)), array)


def stupid_cpu(points):
    l = len(points)
    result = np.zeros(l, dtype=np.float32)
    result[:] = np.inf

    for x in range(l):
        for y in range(l):
            distance = vincenty(points[x, 0], points[x, 1],
                                points[y, 0], points[y, 1])

            if (distance < result[x]):
                result[x] = distance

    return result
