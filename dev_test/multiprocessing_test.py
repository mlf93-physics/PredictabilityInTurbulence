# from multiprocessing import Pool
# import time

# COUNT = 50000000
# def countdown(n):
#     while n>0:
#         n -= 1

# if __name__ == '__main__':
#     pool = Pool(processes=2)
#     start = time.time()
#     r1 = pool.apply_async(countdown, [COUNT//2])
#     r2 = pool.apply_async(countdown, [COUNT//2])
#     pool.close()
#     pool.join()
#     end = time.time()
#     print('Time taken in seconds -', end - start)

import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count


def square(n):

    time.sleep(2)

    return n * n


def main():

    start = timer()

    print(f'starting computations on {cpu_count()} cores')

    values = (2, 4, 6, 8)

    with Pool() as pool:
        res = pool.map(square, values)
        print(res)

    end = timer()
    print(f'elapsed time: {end - start}')

if __name__ == '__main__':
    main()