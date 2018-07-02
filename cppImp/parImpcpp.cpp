#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <iostream>
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
		for(int j=0; j<all_devices.size(); ++j)
			printf("Device %d: %s\n", j, all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
	}

	return all_devices[i];
}

int main() {

	Platform platform = getPlatform();
	Device device = getDevice(platform, 2, true);
}