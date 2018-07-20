
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "error_handler.hpp"
#include "helper_functions.hpp"

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define KERNEL_FILE "findSep3.cl"
#define INPUT_FILE "input2.txt"
#define GLOBAL_SIZE 1024
#define LOCAL_SIZE 64

using namespace std;

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

   ifstream progFile(filename);
   string src(istreambuf_iterator<char>(progFile), (istreambuf_iterator<char>()));

   const char * src_c = src.c_str();
   size_t src_size = src.size();
   

   cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_size, &err);
   error_handler(err, "Couldn't create the program");

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   error_handler(err, "Failed to Build Program");

   return program;*/
}


int main(int argc, char** argv){


   //Get input file
   std::string chunk, residual;
   std::ifstream inputFile(INPUT_FILE);
   read_chunk(inputFile, chunk, residual);

   cl_char* c_chunk = (cl_char*)malloc(chunk.size()*sizeof(cl_char));
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
            CL_MEM_COPY_HOST_PTR, chunk.size()*sizeof(cl_char), c_chunk, &err);

   cl_mem out = clCreateBuffer(context, CL_MEM_READ_WRITE, chunk.size()*sizeof(cl_uint), NULL, &err);
   error_handler(err, "Failed to create out buffer");

   cl_mem out_pos  = clCreateBuffer(context, CL_MEM_READ_WRITE, chunk.size()*sizeof(cl_int), NULL, &err);
   
   cl_uint* pos_ptr = (cl_uint*)malloc(sizeof(cl_uint));
   *pos_ptr = 0;
   cl_mem pos_ptr_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | 
      CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), pos_ptr, &err); 
   error_handler(err);
   
   //Create kernels
   cl_kernel newLine = clCreateKernel(program, "newLine", &err);
   error_handler(err, "Failed to create newLine kernel");
   cl_kernel findSep = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Failed to create findSep kernel");


   

   err = clSetKernelArg(newLine, 0, sizeof(cl_mem), &input_string);
   err = clSetKernelArg(newLine, 1, sizeof(cl_mem), &out);
   err = clSetKernelArg(newLine, 2, sizeof(cl_uint), &chunk_size);
   err = clSetKernelArg(newLine, 3, sizeof(cl_mem), &out_pos);
   err = clSetKernelArg(newLine, 4, sizeof(cl_mem), &pos_ptr_buffer);
   error_handler(err, "Couldn't set args for newLine");

   err = clEnqueueNDRangeKernel(queue, newLine, 1, NULL, &global_size, NULL, 0, NULL, NULL);
   error_handler(err, "Couldn't enqueue kernel");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish1");


   cl_uint * out_num = (cl_uint*)malloc(sizeof(cl_uint));
   

   err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(cl_uint), out_num, 0, NULL, NULL);
   error_handler(err, "Couldn't read buffer");
   
   cl_uint line_num = out_num[0];
   line_num++;
   cl_int * line_out_arr = (cl_int*)malloc(line_num*sizeof(cl_int));

   err = clEnqueueReadBuffer(queue, out_pos, CL_TRUE, 0, line_num*sizeof(cl_int), line_out_arr, 0, NULL, NULL);
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish2");


   vector<cl_int> line_out_vec(line_out_arr, line_out_arr+line_num);
   sort(line_out_vec.begin(), line_out_vec.end());
   for(size_t i=0; i<line_num; ++i) {
      std::cout << line_out_vec[i] << " ";
   }
   cout<<endl<<endl;

   vector<cl_int> input_pos_vec;
   input_pos_vec.push_back(line_out_vec[0]);
   for(uint i = 1; i<line_num; ++i) {
      input_pos_vec.push_back(line_out_vec[i]);
      input_pos_vec.push_back(line_out_vec[i]+1);
   }
   for(size_t i=0; i<(line_num-1)*2; ++i) {
      std::cout << input_pos_vec[i] << " ";
   }
   cout<<endl<<endl;
   cl_int* input_pos = (cl_int*)malloc(input_pos_vec.size()*sizeof(cl_int));
   std::copy(input_pos_vec.begin(), input_pos_vec.end(), input_pos);

   //Buffers for findSep kernel
   cl_mem input_pos_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
                           CL_MEM_COPY_HOST_PTR, input_pos_vec.size()*sizeof(cl_int), input_pos, &err);
   cl_mem final_result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                 chunk.size()*sizeof(cl_uint), NULL, &err);
   cl_mem result_sizes_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                 input_pos_vec.size()*sizeof(cl_uint), NULL, &err);
   error_handler(err, "Failed to create buffers");
   
   cl_uint nlines = input_pos_vec.size()/2;
   cl_mem pos_ptr_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
      CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), pos_ptr, &err); 
   error_handler(err);
   //Set kernel args for findSep
   err = clSetKernelArg(findSep, 0, sizeof(cl_mem), &input_string);//input_string
   err = clSetKernelArg(findSep, 1, sizeof(cl_mem), &input_pos_buffer);//input_pos
   err = clSetKernelArg(findSep, 2, sizeof(cl_mem), &pos_ptr_buffer2);//pos_ptr
   err = clSetKernelArg(findSep, 3, local_size*sizeof(cl_uint), NULL);//separators
   err = clSetKernelArg(findSep, 4, sizeof(cl_mem), &final_result_buffer);//finalResults
   err = clSetKernelArg(findSep, 5, sizeof(cl_mem), &result_sizes_buffer);//result_sizes
   err = clSetKernelArg(findSep, 6, local_size*sizeof(cl_char), NULL);//lstring
   err = clSetKernelArg(findSep, 7, local_size*sizeof(cl_char), NULL);//escape
   err = clSetKernelArg(findSep, 8, local_size*sizeof(cl_char), NULL);//function
   err = clSetKernelArg(findSep, 9, sizeof(cl_uint), &nlines);//lines
   err = clSetKernelArg(findSep, 10, sizeof(cl_uint), NULL);//len
   err = clSetKernelArg(findSep, 11, sizeof(cl_uint), NULL);//curr_pos
   err = clSetKernelArg(findSep, 12, sizeof(cl_char), NULL);//prev_escape
   err = clSetKernelArg(findSep, 13, sizeof(cl_char), NULL);//prev_function
   err = clSetKernelArg(findSep, 14, sizeof(cl_uint), NULL);//prev_sep
   err = clSetKernelArg(findSep, 15, sizeof(cl_char), NULL);//elems_scanned
   err = clSetKernelArg(findSep, 16, sizeof(cl_char), NULL);//first_char
   error_handler(err, "Couldn't set findSep args");

   err = clEnqueueNDRangeKernel(queue, findSep, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
   error_handler(err, "Couldn't enqueue kernel");
   err = clFinish(queue);
   error_handler(err, "Couldn't do clFinish3");

   cl_uint* finalResults = (cl_uint*)malloc(chunk.size()*sizeof(cl_char));
   err = clEnqueueReadBuffer(queue, final_result_buffer, CL_TRUE, 0, chunk.size()*sizeof(cl_char), finalResults, 0, NULL, NULL);

   for(cl_uint i = 0; i<chunk.size(); ++i) {
      cout<<finalResults[i]<<" ";
   }
   cout<<endl<<endl;

   free(pos_ptr);
   free(out_num);
   free(line_out_arr);
   free(finalResults);

   clReleaseMemObject(input_string);
   clReleaseMemObject(out);
   clReleaseMemObject(input_pos_buffer);
   clReleaseMemObject(final_result_buffer);
   clReleaseMemObject(result_sizes_buffer);
   clReleaseMemObject(pos_ptr_buffer);
   clReleaseMemObject(pos_ptr_buffer2);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseDevice(device);
   clReleaseContext(context);

   return 0;
}