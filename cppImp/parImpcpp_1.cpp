
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "error_handler.hpp"
#include "helper_functions.hpp"

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define KERNEL_FILE "test.cl"
#define INPUT_FILE "input.txt"

cl_device_id create_device() {
   cl_int err;

   cl_uint num_plats;
   clGetPlatformIDs(0, NULL, &num_plats);
   if(num_plats == 0){
      printf("No platforms found\n");
      exit(-1);
   }
   cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_plats);
   err = clGetPlatformIDs(num_plats, platforms, NULL);
   error_handler(err, "Couldn't get platform");

   cl_device_id device;
   for(cl_uint i=0; i<num_plats; ++i){
      err = clGetDeviceIDs(platforms[i], DEVICE_TYPE, 1, &device, NULL);
      if(err < 0) continue;
      cl_bool canUse;
      clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &canUse, NULL);
      if(canUse){
         break;
      }
   }
   error_handler(err, "Couldn't access any devices of specified type");

   return device;
}

cl_program build_program(cl_context context, std::string filename){
   cl_int err;

   std::ifstream progFile(filename);
   std::string src(std::istreambuf_iterator<char>(progFile), (std::istreambuf_iterator<char>()));

   const char * src_c = src.c_str();
   size_t src_size = src.size();
   

   cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_size, &err);
   error_handler(err, "Couldn't create the program");

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   error_handler(err, "Failed to Build Program");

   return program;
}

int main(){

   cl_int err;

   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   cl_program program = build_program(context, KERNEL_FILE);

   // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
   cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
   error_handler(err, "Failed to create command queue");

   cl_kernel initFunc = clCreateKernel(program, "initFunc", &err);
   error_handler(err, "Failed to create initFunc kernel");

   size_t global_size = 1024, local_size = 64;

   cl_mem out = clCreateBuffer(context, CL_MEM_READ_WRITE, global_size*sizeof(cl_uint), NULL, &err);
   error_handler(err, "Failed to create out buffer");

   err = clSetKernelArg(initFunc, 0, sizeof(cl_mem), &out);
   error_handler(err, "Couldn't set kernel arg");
   err = clSetKernelArg(initFunc, 1, sizeof(cl_uint), &global_size);
   error_handler(err, "Couldn't set kernel arg");

   err = clEnqueueNDRangeKernel(queue, initFunc, 1, NULL, &global_size, NULL, 0, NULL, NULL);
   error_handler(err, "Couldn't enqueue kernel");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish");


   cl_uint * out_arr = (cl_uint*)malloc(global_size*sizeof(cl_uint));

   err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, global_size*sizeof(cl_uint), out_arr, 0, NULL, NULL);
   error_handler(err, "Couldn't read buffer");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish");

   for(size_t i=0; i<global_size; ++i){
      std::cout << out_arr[i] << " ";
   }
   std::cout << std::endl;

   free(out_arr);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseDevice(device);
   clReleaseContext(context);

   return 0;
}