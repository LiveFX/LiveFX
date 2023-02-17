# def gaussian_kernel(size, sigma):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g


# def gaussian_blur(frame, x, y, x_end, y_end, kernel_size=5, sigma=0.5):
#     blurred_frame = frame.copy()
#     kernel = gaussian_kernel(kernel_size, sigma)
#     kernel_size = kernel.shape[0]
#     size = kernel_size // 2
#     rows, cols, channels = frame.shape
#     for channel in range(channels):
#         for row in range(size, rows-size):
#             for col in range(size, cols-size):
#                 if row > y and row < y_end and col > x and col < x_end:
#                     continue
#                 else:
#                     region = frame[row-size:row+size +
#                                    1, col-size:col+size+1, channel]
#                     blurred_frame[row, col, channel] = (region * kernel).sum()
#     return blurred_frame


# import cupy as cp

# def gaussian_kernel(size, sigma):
#     size = int(size) // 2
#     x, y = cp.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * cp.pi * sigma**2)
#     g = cp.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g


# def gaussian_blur_gpu(frame, x, y, x_end, y_end, kernel_size=5, sigma=0.5):
#     frame_gpu = cp.asarray(frame)
#     blurred_frame = frame_gpu.copy()
#     kernel = gaussian_kernel(kernel_size, sigma)
#     kernel_size = kernel.shape[0]
#     size = kernel_size // 2
#     rows, cols, channels = frame_gpu.shape
#     for channel in range(channels):
#         for row in range(size, rows-size):
#             for col in range(size, cols-size):
#                 if row > y and row < y_end and col > x and col < x_end:
#                     continue
#                 else:
#                     region = frame_gpu[row-size:row+size +
#                                    1, col-size:col+size+1, channel]
#                     blurred_frame[row, col, channel] = (region * kernel).sum()
#     return cp.asnumpy(blurred_frame)


# import concurrent.futures

# def gaussian_kernel(size, sigma):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g

# def apply_filter(frame,kernel, x, y, x_end, y_end, kernel_size = 5, sigma = 0.5):
#     blurred_frame = frame.copy()
#     size = kernel_size // 2
#     rows, cols, channels = frame.shape
#     for channel in range(channels):
#         for row in range(size, rows-size):
#             for col in range(size, cols-size):
#                 if row > y and row < y_end and col > x and col < x_end:
#                     continue
#                 else:
#                     region = frame[row-size:row+size+1, col-size:col+size+1,channel]
#                     blurred_frame[row, col,channel] = (region * kernel).sum()
#     return blurred_frame

# def gaussian_blur(frame, x, y, x_end, y_end, kernel_size = 5, sigma = 0.5, num_cores = 4):
#     blurred_frame = frame.copy()
#     kernel = gaussian_kernel(kernel_size, sigma)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
#         splitted_frame = np.array_split(frame, num_cores)
#         results = [executor.submit(apply_filter, split_frame, kernel, x, y, x_end, y_end, kernel_size, sigma) for split_frame in splitted_frame]
#         for future in concurrent.futures.as_completed(results):
#             split_result = future.result()
#             blurred_frame[split_result[0]:split_result[1],:,:] = split_result[2]
#     return blurred_frame
