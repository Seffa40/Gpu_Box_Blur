
import cv2
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

cuda.init()

device = cuda.Device(0)
context = device.make_context()

print("Libraries loaded and CUDA context successfully initialized.")

def load_image_to_gpu(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print("Image Dimensions:", image.shape)

    image_uint8 = image.astype(np.uint8)
    print(f"Image Memory Size (nbytes): {image_uint8.nbytes}")

    image_gpu = cuda.mem_alloc(image_uint8.nbytes)
    cuda.memcpy_htod(image_gpu, image_uint8)
    return image, image_gpu, image_uint8

def retrieve_image_from_gpu(image_gpu, image_shape):
    output_image = np.empty(image_shape, dtype=np.uint8)
    cuda.memcpy_dtoh(output_image, image_gpu)
    return output_image

kernel_code = """
__global__ void box_blur(unsigned char* img, unsigned char* out_img, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum_r = 0, sum_g = 0, sum_b = 0;
        int count = 0;

        // Traverse a 3x3 window and average the neighboring pixels
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);

                // Add R, G, B components
                int idx = (ny * width + nx) * 3;
                sum_r += img[idx];
                sum_g += img[idx + 1];
                sum_b += img[idx + 2];
                count++;
            }
        }

        // Compute average value and write it
        int idx_out = (y * width + x) * 3;
        out_img[idx_out] = sum_r / count;
        out_img[idx_out + 1] = sum_g / count;
        out_img[idx_out + 2] = sum_b / count;
    }
}
"""

mod = SourceModule(kernel_code)
box_blur_kernel = mod.get_function("box_blur")
print("Box blur CUDA kernel function successfully defined and compiled.")

image_path = r'image.jpg'
image, image_gpu, image_uint8 = load_image_to_gpu(image_path)

block_size = (16, 16, 1)
grid_size = (
    int(np.ceil(image.shape[1] / block_size[0])),
    int(np.ceil(image.shape[0] / block_size[1]))
)
output_image_gpu = cuda.mem_alloc(image_uint8.nbytes)

print(f"Block Size: {block_size}")
print(f"Grid Size: {grid_size}")

start_time = time.time()

box_blur_kernel(
    image_gpu, output_image_gpu,
    np.int32(image.shape[1]), np.int32(image.shape[0]),
    block=block_size, grid=grid_size
)

cuda.Context.synchronize()

gpu_time = time.time() - start_time
print(f"GPU Execution Time: {gpu_time} seconds")

output_image = retrieve_image_from_gpu(output_image_gpu, image.shape)
print("Data successfully retrieved from GPU.")

start_time = time.time()

cpu_output = cv2.blur(image, (3, 3))

cpu_time = time.time() - start_time
print(f"CPU Execution Time: {cpu_time} seconds")

cv2.putText(image, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(output_image, 'GPU Box Blur', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(cpu_output, 'CPU Box Blur', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Original Image', image)
cv2.imshow('GPU Blurred Image', output_image)
cv2.imshow('CPU Blurred Image', cpu_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"GPU vs CPU Execution Time Comparison:")
print(f"GPU Time: {gpu_time} seconds, CPU Time: {cpu_time} seconds")

output_image_gpu.free()
image_gpu.free()
print("GPU memory released.")

context.pop()
print("CUDA context terminated.")
