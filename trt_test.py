import numpy as np
import cv2
import timeit
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


weight_paths = "/vscode/sample_onnx/pretrained/mnist.plan"
trt_logger = trt.Logger(trt.Logger.INFO)
bindings = []
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
with open(weight_paths, 'rb') as f, trt.Runtime(trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
stream = cuda.Stream()

input_volume = trt.volume((1, 28, 28))
batch_size = 1
max_batch_size = engine.max_batch_size
# print(max_batch_size)
# raise SystemExit
numpy_array = np.zeros((max_batch_size, input_volume))

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()

# img = cv2.imread("data/BruceLee.jpg")
# img = cv2.resize(img, (112, 112))
img = np.random.randn(1, 28, 28).astype(np.float32)
# img = img.astype(np.float32) / 255.
# image_tensor = np.stack([img] * 1)

while True:
    for i in range(batch_size):
        numpy_array[i] = img.ravel()

    np.copyto(host_inputs[0], numpy_array.ravel())
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    t0 = timeit.default_timer()
    context.execute_async(
        batch_size=max_batch_size,
        bindings=bindings,
        stream_handle=stream.handle)

    for i in range(len(host_outputs)):
        cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
    stream.synchronize()

    print("{:.5f}".format(timeit.default_timer() - t0))
    print(host_outputs[0].shape)