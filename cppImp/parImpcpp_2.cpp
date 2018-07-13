
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

#include "error_handler.hpp"
#include "helper_functions.hpp"

void getPlatform() {
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
         break;
		}
	}
	cl::Platform newP = cl::Platform::setDefault(plat);
	if(newP != plat) {
		std::cout<<"Error setting default platform.";
		exit(1);
	}
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

   cl_int err;

   std::cout << "-2" << std::endl;

	getPlatform();
	cl::Platform platform = cl::Platform::getDefault(&err);
	error_handler(err);
	cl::Device device = getDevice(platform, 0, true);
	cl::Device newD = cl::Device::setDefault(device);
	if(newD != device) {
		std::cout<<"Error setting default device.";
		return -1;
	}
	//device = cl::Device::getDefault(&err);
	//error_handler(err);

   std::cout << "-1" << std::endl;
   
	// Select the default platform and create a context using this platform and the GPU
   cl_context_properties cps[3] = { 
      CL_CONTEXT_PLATFORM, 
      (cl_context_properties)(platform)(), 
      0 
   };
   cl::Context context( device, cps, NULL, NULL, &err);
   error_handler(err);

   std::cout << "0" << std::endl;

   // Create a command queue and use the first device
   cl::CommandQueue queue(context, device);

   // Read source file
   std::ifstream sourceFile("findSepNew.cl");
   
   std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
   //std::cout << sourceCode << std::endl;

   std::vector<std::string> programString {sourceCode};
   cl::Program program(programString);
   try {
   	program.build("-cl-std=CL2.0");
   }
   catch (...) {
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		for (auto &pair : buildInfo) {
			std::cerr << pair.second << std::endl << std::endl;
		}
		return 1;
	}

	cl::KernelFunctor<> newLine(program, "newLine", &err);
	error_handler(err);

   /*cl::Program::Sources sources;

   sources.push_back({sourceCode.c_str(), sourceCode.length()+1});

   // Make program of the source code in the context
   cl::Program program = cl::Program(context, sources, &err);
   error_handler(err);

   std::cout << "1" << std::endl;

   //Needs to be vector of devices
   err = program.build({device});
   error_handler(err);*/
   
   std::cout << "2" << std::endl;

   unsigned int global_size,
                local_size = 1024;/*

   std::cout << "3" << std::endl;

   cl::Kernel newLine(program, "newLine", &err);
   error_handler(err);
   cl::Kernel findSep(program, "findSep");*/

   std::string chunk, residual;

   std::ifstream inputFile("input.txt");
   read_chunk_pp(inputFile, chunk, residual);

   global_size = chunk.size();
   cl_char * c_chunk = (cl_char *)malloc(chunk.size());
   for(unsigned int i=0; i<chunk.size(); ++i){
      c_chunk[i] = chunk[i];
      std::cout << c_chunk[i];
   }
   std::cout << "\n" << std::endl;

   std::cout << "4" << std::endl;

   /*cl::Buffer input_string(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           chunk.size(), c_chunk);

   cl::Buffer newLine_arr(context, CL_MEM_READ_WRITE, sizeof(int)*chunk.size());

   unsigned int * output = (unsigned int *)calloc(chunk.size(), sizeof(unsigned int));
   // unsigned int * output = new unsigned int[chunk.size()];

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
      std::cout << output[i];
   }
   std::cout << std::endl;

   free(output);*/
   // delete[] output;

   return 0;
}