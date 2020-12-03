#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
    }

    ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);

    if (!onnxFile.good()) {
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();
}