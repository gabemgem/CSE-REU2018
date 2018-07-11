
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "error_handler.hpp"
#include "helper_functions.hpp"

cl::Platform getPlatform() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if(all_platforms.size()==0) {
		std::cout << "No platforms found.\n";
		exit(1);
	}
	cl::Platform plat;
	for(auto &p : all_platforms) {
		std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if(platver.find("OpenCL 2.") != std::string::npos) {
			plat = p;
		}
	}
	return plat;
}

cl::Device getDevice(cl::Platform platform, int i, bool display=false) {
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if(all_devices.size()==0) {
		std::cout << "No devices found.\n";
		exit(1);
	}

	if(display) {
		for(auto &d : all_devices){
			std::cout << "Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
      }
	}

	return all_devices[i];
}

int main() {
   
   std::cout << "-2" << std::endl;

	cl::Platform platform = getPlatform();
	cl::Device device = getDevice(platform, 0);

   std::cout << "-1" << std::endl;

	// Select the default platform and create a context using this platform and the GPU
   cl_context_properties cps[3] = { 
      CL_CONTEXT_PLATFORM, 
      (cl_context_properties)(platform)(), 
      0 
   };
   cl::Context context( CL_DEVICE_TYPE_CPU, cps);

   std::cout << "0" << std::endl;

   // Create a command queue and use the first device
   cl::CommandQueue queue(context, device);

   // Read source file
   std::ifstream sourceFile("findSep.cl");
   
   std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

   cl::Program::Sources sources;

   sources.push_back({sourceCode.c_str(), sourceCode.length()+1});

   // Make program of the source code in the context
   cl::Program program = cl::Program(context, sources);

   std::cout << "1" << std::endl;

   //Needs to be vector of devices
   program.build({device});
   
   std::cout << "2" << std::endl;

   unsigned int global_size = 1 << 31,
                local_size = 1024;

   std::cout << "3" << std::endl;

   cl::Kernel newLine(program, "newLine");
   cl::Kernel findSep(program, "findSep");

   std::string chunk, residual;

   std::ifstream inputFile("input.txt");
   read_chunk_pp(inputFile, chunk, residual);

   std::cout << "4" << std::endl;

   cl::Buffer input_string(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           chunk.size(), (void *)chunk.c_str());

   cl::Buffer newLine_arr(context, CL_MEM_READ_WRITE, sizeof(int)*chunk.size());

   unsigned int * output = (unsigned int *)malloc(chunk.size());

   std::cout << "5" << std::endl;

   newLine.setArg(0, input_string);
   newLine.setArg(1, newLine_arr);
   newLine.setArg(2, chunk.size());

   std::cout << "5" << std::endl;

   queue.enqueueNDRangeKernel(newLine, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
   queue.finish();

   std::cout << "6" << std::endl;

   queue.enqueueReadBuffer(newLine_arr, CL_TRUE, 0, sizeof(int)*chunk.size(), output);

   std::cout << "7" << std::endl;

   for(unsigned int i=0; i<chunk.size(); ++i){
      if(output[i]){
         std::cout << i << " ";
      }
   }
   std::cout << std::endl;

   free(output);

   return 0;
}