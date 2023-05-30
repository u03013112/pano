import numpy as np
import timeit

def measure_lonlat2XY():
    lonlat = np.random.rand(1000, 1000, 2)
    shape = (1000, 1000)

    start_time = timeit.default_timer()

    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)

    elapsed_time = timeit.default_timer() - start_time
    print(f"Execution time: {elapsed_time * 1000:.2f} ms")

measure_lonlat2XY()
