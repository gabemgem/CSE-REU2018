#define VERBOSE 0

#define INPUT_FILE "input.txt"
#define PROGRAM_FILE "findSep.cl"
//#define INPUT_SIZE 64//Use if input size is already known
#define _GNU_SOURCE
#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "error_handler.h"

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   error_handler(err, "Couldn't identify a platform");

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   error_handler(err, "Couldn't access any devices");

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
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
   program = clCreateProgramWithSource(ctx, 1, 
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
}

void read_from_file(char* line, cl_int len) {
   FILE *fp;

   fp = fopen(INPUT_FILE, "r");
   if(!fp) {
      printf("Couldn't open input file");
      exit(1);
   }

   if(fgets(line, len, fp) == NULL) {
      printf("Couldn't read input file");
      exit(1);
   }

   fclose(fp);

}

/* Padding string w/ spaces to a length of next power of 2.
   Stores padded string and new length in parameters.
*/
void pad_string(char** str, cl_int* len){ 
   //probably a faster implementation
   cl_int new_len = 1;

   while(*len > new_len){
      new_len <<= 1;
   }

   *str = (char *)realloc(*str, sizeof(char) * new_len);
   
   for(cl_int i=(*len); i<new_len; ++i){
         (*str)[i] = ' ';
   }
   *len = new_len;
}

cl_uint lg(int val){
   cl_uint out = 0;
   while(val > 1){
      val >>= 1;
      ++out;
   }
   return out;
}

int main(int argc, char** argv) {

   if(argc!=2) {
      perror("Incorrect number of arguments");
      perror("Pass in string length");
      exit(1);
   }
   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;

   cl_kernel initFunction;
   cl_kernel scanStep;
   cl_kernel addScanStep;
   cl_kernel postScanIncStep;
   cl_kernel addPostScanIncStep;
   cl_kernel findSep;
   cl_kernel compressRes;
   
   cl_command_queue queue;
   cl_int err;

   /* Data and buffers */
   cl_int input_length = atoi(argv[1]) + 1;
   char* input_string = malloc(input_length * sizeof(char));

   /* Get data from file */
   read_from_file(input_string, input_length);
   
   /* Pads string to length of next power of 2 with spaces */
   pad_string(&input_string, &input_length);

   if(VERBOSE){
      printf("%d\n", input_length);
   }

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create work group size */
   size_t global_size = input_length;//Total NUM THREADS
   size_t local_size = 8;//NUM THREADS per BLOCK
   
   /* Shared memory for findSep */
   cl_char firstCharacter = (input_string[0] == SEP);
   cl_uint* finalResults = malloc(input_length * sizeof(cl_uint));
   
   /* Create buffers */
   cl_mem input_buffer, function_buffer, output_buffer, escape_buffer;

   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, input_length * sizeof(char), input_string, &err);
   function_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                     input_length * sizeof(cl_char), NULL, &err);
   output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                   input_length * sizeof(cl_uint), NULL, &err);
   escape_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                   input_length * sizeof(cl_char), NULL, &err);

   error_handler(err, "Couldn't create buffers");

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   error_handler(err, "Couldn't create a command queue");

   /* Creating kernels */
   initFunction = clCreateKernel(program, "initFunc", &err);
   error_handler(err, "Couldn't create initFunc kernel");

   scanStep = clCreateKernel(program, "scanStep", &err);
   error_handler(err, "Couldn't create scanStep kernel");

   postScanIncStep = clCreateKernel(program, "postScanIncStep", &err);
   error_handler(err, "Couldn't create postScanIncStep kernel");

   addScanStep = clCreateKernel(program, "addScanStep", &err);
   error_handler(err, "Couldn't create addScanStep kernel");

   addPostScanIncStep = clCreateKernel(program, "addPostScanIncStep", &err);
   error_handler(err, "Couldn't create addPostScanIncStep kernel");

   findSep = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Couldn't create findSep kernel");

   // Setting up and running init function
   err = clSetKernelArg(initFunction, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(initFunction, 1, sizeof(cl_int), &input_length);
   err |= clSetKernelArg(initFunction, 2, sizeof(cl_mem), &escape_buffer);
   err |= clSetKernelArg(initFunction, 3, sizeof(cl_mem), &function_buffer);
   error_handler(err, "Couldn't create a kernel argument for initFunction");
   
   err = clEnqueueNDRangeKernel(queue, initFunction, 1, NULL, &global_size, 
      &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the initFunc kernel");

   clFinish(queue);
   if(VERBOSE){
      printf("Finished init\n");
   }

   cl_uint depth = lg(input_length), d;
   for(d=0; d<depth; ++d){
      err = clSetKernelArg(scanStep, 0, sizeof(cl_mem), &function_buffer);
      err |= clSetKernelArg(scanStep, 1, sizeof(cl_uint), &input_length);
      err |= clSetKernelArg(scanStep, 2, sizeof(cl_uint), &d);
      error_handler(err, "Couldn't create a kernel argument for scanStep");

      err = clEnqueueNDRangeKernel(queue, scanStep, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
      error_handler(err, "Couldn't enqueue the scanStep");
   
      //clFinish(queue);
   }

   if(VERBOSE){
      printf("Finished scan step\n");
   }

   cl_uint stride;
   for(stride = input_length/4; stride > 0; stride /= 2){
      err = clSetKernelArg(postScanIncStep, 0, sizeof(cl_mem), &function_buffer);
      err |= clSetKernelArg(postScanIncStep, 1, sizeof(cl_uint), &input_length);
      err |= clSetKernelArg(postScanIncStep, 2, sizeof(cl_uint), &stride);
      error_handler(err, "Couldn't create a kernel argument for postScanIncStep");

      err = clEnqueueNDRangeKernel(queue, postScanIncStep, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
      error_handler(err, "Couldn't enqueue the postScanIncStep");
   
      //clFinish(queue);
   }
   
   if(VERBOSE){
      printf("Finished post scan step\n");
   }

   err = clSetKernelArg(findSep, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(findSep, 1, sizeof(cl_uint), &input_length);
   err |= clSetKernelArg(findSep, 2, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(findSep, 3, sizeof(cl_char), &firstCharacter);
   err |= clSetKernelArg(findSep, 4, sizeof(cl_mem), &output_buffer);
   error_handler(err, "Couldn't create a kernel argument for findSep");

   err = clEnqueueNDRangeKernel(queue, findSep, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the findSep kernel");

   clFinish(queue);

   if(VERBOSE){
      printf("Finished separation\n");
   }

   for(d=0; d<depth; ++d){
      err = clSetKernelArg(addScanStep, 0, sizeof(cl_mem), &output_buffer);
      err |= clSetKernelArg(addScanStep, 1, sizeof(cl_uint), &input_length);
      err |= clSetKernelArg(addScanStep, 2, sizeof(cl_uint), &d);
      error_handler(err, "Couldn't create a kernel argument for addScanStep");

      err = clEnqueueNDRangeKernel(queue, addScanStep, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
      error_handler(err, "Couldn't enqueue the addScanStep");
   
      //clFinish(queue);
   }

   if(VERBOSE){
      printf("Finished add scan step\n");
   }

   for(stride = input_length/4; stride > 0; stride /= 2){
      err = clSetKernelArg(addPostScanIncStep, 0, sizeof(cl_mem), &output_buffer);
      err |= clSetKernelArg(addPostScanIncStep, 1, sizeof(cl_uint), &input_length);
      err |= clSetKernelArg(addPostScanIncStep, 2, sizeof(cl_uint), &stride);
      error_handler(err, "Couldn't create a kernel argument for addPostScanIncStep");

      err = clEnqueueNDRangeKernel(queue, addPostScanIncStep, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
      error_handler(err, "Couldn't enqueue the addPostScanIncStep");
   
      //clFinish(queue);
   }
   if(VERBOSE){
      printf("Finished add post scan step\n");
   }
   clFinish(queue);
   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), finalResults, 0, NULL, NULL);
   error_handler(err, "Couldn't read the buffer");

   cl_uint num = finalResults[input_length-1];
   cl_mem compressedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                   num * sizeof(cl_uint), NULL, &err);

   compressRes = clCreateKernel(program, "compressResults", &err);
   error_handler(err, "Couldn't create compressRes kernel");
   
   err = clSetKernelArg(compressRes, 0, sizeof(cl_mem), &output_buffer);
   err |= clSetKernelArg(compressRes, 1, sizeof(cl_mem), &compressedBuffer);
   error_handler(err, "Couldn't create a kernel argument for compressRes");
   
   err = clEnqueueNDRangeKernel(queue, compressRes, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
      error_handler(err, "Couldn't enqueue the compressRes");
   clFinish(queue);

   cl_uint* compressedResults = malloc(num * sizeof(cl_uint));
   err = clEnqueueReadBuffer(queue, compressedBuffer, CL_TRUE, 0,
         num * sizeof(cl_uint), compressedResults, 0, NULL, NULL);
   error_handler(err, "Couldn't read the compressed buffer");

   printf("%s\n", input_string);
   for(int i=0; i<input_length; ++i) {
      printf("%d", finalResults[i]);
   }
   printf("\n");
   for(int i=0; i<num; ++i) {
      printf("%d, ", compressedResults[i]);
   }
   printf("\n");
   
   /* Deallocate resources */
   free(input_string);
   free(finalResults);

   clReleaseDevice(device);

   clReleaseKernel(initFunction);
   clReleaseKernel(scanStep);
   clReleaseKernel(postScanIncStep);
   clReleaseKernel(findSep);

   clReleaseMemObject(input_buffer);
   clReleaseMemObject(function_buffer);
   clReleaseMemObject(output_buffer);
   clReleaseMemObject(escape_buffer);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
