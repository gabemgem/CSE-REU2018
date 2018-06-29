
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cl;

Platform getPlatform() {
	vector<Platform> all_platforms;
	Platform::get(&all_platforms);

	if(all_platforms.size()==0) {
		cout << "No platforms found.\n";
		exit(1);
	}
	Platform plat;
	for(auto &p : platforms) {
		string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if(platver.find("OpenCL 1.2") != string::npos) {
			plat = p;
		}
	}
	return plat;
}

Device getDevice(Platform platform, int i, bool display=false) {
	vector<Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0) {
		cout << "No devices found.\n";
		exit(1);
	}

	if(display) {
		for(int j=0; j<all_devices.size(); ++j)
			printf("Device %d: %s\n", j, all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
	}

	return all_devices[i];
}

int main() {

	Platform platform = getPlatform();
	Device device = getDevice(platform, 2, true);

	// Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = { 
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)(platform)(), 
        0 
    };
    Context context( CL_DEVICE_TYPE_GPU, cps);


    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, device);

    // Read source file
    ifstream sourceFile("findSep.cl");
    string sourceCode(
        istreambuf_iterator<char>(sourceFile),
        (istreambuf_iterator<char>()));
    Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // Make program of the source code in the context
    Program program = Program(context, source);

    program.build(device);

    Kernel initFunc(program, "initFunc");
}