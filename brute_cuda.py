import math
import numpy as np
from numba import cuda
from cuda_friendly_vincenty import vincenty


wrap = cuda.jit('float32(float32, float32, float32, float32)', device=True)
vincenty = wrap(vincenty)


@cuda.jit('void(float32[:,:], float32[:])')
def find_all(arr, res):
    x, y = cuda.grid(2)

    if x > y and (x < res.shape[0]) and (y < res.shape[0]):
        dist = vincenty(
                arr[x, 0], arr[x, 1],
                arr[y, 0], arr[y, 1])

        cuda.atomic.min(res, x, dist)
        cuda.atomic.min(res, y, dist)


def brute_cuda(array):
    n = len(array)

    result = np.zeros(n, dtype=np.float32)
    result[:] = np.inf

    device_array = cuda.to_device(array)
    device_result = cuda.to_device(result)

    # Configure the blocks
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(n / threadsperblock[1])
    blockspergrid_y = math.ceil(n / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    find_all[blockspergrid, threadsperblock](device_array, device_result)

    return device_result.copy_to_host()
