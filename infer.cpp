#include <iostream>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <memory>
#include <math.h>
#include <iomanip>

#include <cuda_runtime.h>

#include "engine.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc!=3) {
		cerr << "Usage: " << argv[0] << " engine.plan image.pgm" << endl;
		return 1;
	}

    cout << "Loading engine..." << endl;
    auto engine = sample_onnx::Engine(argv[1]);

    // Create device buffers
    void *data_d, *output_d;
    auto data = engine.processInput(string(argv[2]));
    cudaMalloc(&data_d, 28 * 28 * sizeof(float));
	cudaMalloc(&output_d, 10 * sizeof(float));

    // Copy image to device
	size_t dataSize = data.size() * sizeof(float);
	cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

    // Run inference n times
	cout << "Running inference..." << endl;
    const int count = 1;
	auto start = chrono::steady_clock::now();
 	vector<void *> buffers = { data_d, output_d };
	for (int i = 0; i < count; i++) {
		engine.infer(buffers, 1);
	}
    auto stop = chrono::steady_clock::now();
	auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
	cout << "Took " << timing.count() / count << " seconds per inference." << endl;

    cudaFree(data_d);

	// Get back the results
	unique_ptr<float[]> output(new float[10]);
	cudaMemcpy(output.get(), output_d, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	cudaFree(output_d);

    // Calculate Softmax
    float sum{0.0f};
    float val{0.0f};
    int idx{0};
    for (int i = 0; i < 10; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    cout << "Output:" << endl;
    for (int i = 0; i < 10; i++)
    {
        output[i] /= sum;
        val = max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        cout << " Prob " << i << "  " << fixed << setw(5) << setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << string(int(floor(output[i] * 10 + 0.5f)), '*') << endl;
    }

    return 0;
    
}