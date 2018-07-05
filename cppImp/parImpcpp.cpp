
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cl;

Platform getPlatform() {
	std::vector<Platform> all_platforms;
	Platform::get(&all_platforms);

	if(all_platforms.size()==0) {
		cout << "No platforms found.\n";
		exit(1);
	}
	Platform plat;
	for(auto &p : all_platforms) {
		string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if(platver.find("OpenCL 1.2") != string::npos) {
			plat = p;
		}
	}
	return plat;
}

Device getDevice(Platform platform, int i, bool display=false) {
	std::vector<Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0) {
		cout << "No devices found.\n";
		exit(1);
	}

	if(display) {
		for(auto &d : all_devices){
			cout << "Device " << j << ": " << d.getInfo<CL_DEVICE_NAME>() << endl;
      }
	}

	return all_devices[i];
}

int main() {

	Platform platform = getPlatform();
	Device device = getDevice(platform, 2);

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
   std::ifstream sourceFile("findSep.cl");
   std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));

   Program::Sources sources;

   sources.push_back({sourceCode.c_str(), sourceCode.length()+1});

   // Make program of the source code in the context
   Program program = Program(context, sources);

   //Needs to be vector of devices
   program.build(device);

   Kernel initFunc(program, "initFunc");
}