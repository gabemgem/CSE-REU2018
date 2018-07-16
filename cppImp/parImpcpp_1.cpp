
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "error_handler.hpp"
#include "helper_functions.hpp"

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define KERNEL_FILE "findSepNew.cl"
#define INPUT_FILE "input.txt"
#define GLOBAL_SIZE 1024
#define LOCAL_SIZE 64

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

cl_program build_program(cl_context context, cl_device_id dev, std::string filename){
   
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename.c_str(), "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(context, 1, 
      (const char**)&program_buffer, &program_size, &err);
   error_handler(err, "Couldn't create the program");
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;

   /*
   cl_int err;

   std::ifstream progFile(filename);
   std::string src(std::istreambuf_iterator<char>(progFile), (std::istreambuf_iterator<char>()));

   const char * src_c = src.c_str();
   size_t src_size = src.size();
   

   cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_size, &err);
   error_handler(err, "Couldn't create the program");

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   error_handler(err, "Failed to Build Program");

   return program;*/
}

int main(int argc, char** argv){

   if(argc!=2) {
      std::cout<<"Invalid number of inputs.\n";
      std::cout<<"Please enter in the number of lines\n";
      exit(1);
   }

   cl_uint nlines = atoi(argv[1]);

   //Get input file
   std::string chunk, residual;
   std::ifstream inputFile(INPUT_FILE);
   read_chunk_pp(inputFile, chunk, residual);

   cl_char* c_chunk = (cl_char*)malloc(chunk.size());
   for(unsigned int i=0; i<chunk.size(); ++i){
      c_chunk[i] = chunk[i];
      std::cout<<c_chunk[i];
   }
   std::cout<<std::endl;
   std::cout<<"Read in file\n"<<std::endl;


   size_t global_size = GLOBAL_SIZE;
   size_t local_size = LOCAL_SIZE;
   cl_uint chunk_size = chunk.size();
   cl_int err;


   //Create device, context, program, and queue
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   cl_program program = build_program(context, device, KERNEL_FILE);

   std::cout<<"Built program"<<std::endl;
   // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
   cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
   error_handler(err, "Failed to create command queue");


   //Create buffers
   cl_mem input_string = clCreateBuffer(context, CL_MEM_READ_WRITE |
            CL_MEM_COPY_HOST_PTR, chunk.size(), c_chunk, &err);

   cl_mem out = clCreateBuffer(context, CL_MEM_READ_WRITE, chunk.size()*sizeof(cl_uint), NULL, &err);
   error_handler(err, "Failed to create out buffer");

   
   //Create kernels
   cl_kernel newLine = clCreateKernel(program, "newLine", &err);
   error_handler(err, "Failed to create newLine kernel");


   

   err = clSetKernelArg(newLine, 0, sizeof(cl_mem), &input_string);
   err = clSetKernelArg(newLine, 1, sizeof(cl_mem), &out);
   err = clSetKernelArg(newLine, 2, sizeof(cl_uint), &chunk_size);
   error_handler(err, "Couldn't set args for newLine");

   err = clEnqueueNDRangeKernel(queue, newLine, 1, NULL, &global_size, NULL, 0, NULL, NULL);
   error_handler(err, "Couldn't enqueue kernel");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish");


   cl_uint * out_arr = (cl_uint*)malloc(chunk.size()*sizeof(cl_uint));

   err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, chunk.size()*sizeof(cl_uint), out_arr, 0, NULL, NULL);
   error_handler(err, "Couldn't read buffer");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish");

   for(size_t i=0; i<chunk.size(); ++i){
      std::cout << out_arr[i] << " ";
   }
   std::cout << std::endl;

   free(out_arr);

   clReleaseMemObject(input_string);
   clReleaseMemObject(out);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseDevice(device);
   clReleaseContext(context);

   return 0;
}