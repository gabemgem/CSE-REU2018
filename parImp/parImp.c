#define INPUT_FILE "input.txt"
#define PROGRAM_FILE "findSep.cl"
//#define INPUT_SIZE 64//Use if input size is already known
#define _GNU_SOURCE
#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

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
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
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

void read_from_file(char* line, cl_int* len) {
   FILE *fp;
   line = NULL;
   size_t l = 0;

   fp = fopen(INPUT_FILE, "r");
   if(!fp) {
      printf("Couldn't open input file");
      exit(1);
   }

   if((*len = getline(&line, &l, fp)) == -1) {
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

int main() {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel initFunction;
   cl_kernel parScanFunction;
   cl_kernel parScanFunctionWithSubarrays;
   cl_kernel parScanComposeFuncInc;
   cl_kernel findSeparators;
   cl_command_queue queue;
   cl_int err;

   /* Data and buffers */
   char* input_string;
   cl_int input_length;


   /* Get data from file */
   read_from_file(input_string, &input_length);
   
   //pads string to length of next power of 2 with spaces
   pad_string(&input_string, &input_length);

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create work group size */
   size_t global_size = 128;//Total NUM THREADS
   size_t local_size = 8;//NUM THREADS per BLOCK
   cl_int num_groups = global_size/local_size;//NUM BLOCKS
   
   /* Shared memory for parallel scan kernels */
   //cl_uint* local_array;
   /* Shared memory for findSep */
   cl_char firstCharacter = (input_string[0] == SEP);
   cl_uint* finalResults = malloc(input_length * sizeof(cl_uint));
   
   
   /* Create buffers */
   cl_mem input_buffer, function_buffer, output_buffer;
   cl_mem escape_buffer, partial_buffer, separator_buffer;

   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, input_length * sizeof(char), input_string, &err);

   function_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                     input_length * sizeof(cl_char), NULL, &err);

   output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                   input_length * sizeof(cl_uint), NULL, &err);

   escape_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                   input_length * sizeof(cl_char), NULL, &err);

   partial_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    num_groups * sizeof(cl_uint), NULL, &err);

   separator_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                      input_length * sizeof(cl_uint), NULL, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create input and output buffers");
      exit(1);   
   }

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a command queue");
      exit(1);   
   }

   /* Creating kernels */
   initFunction = clCreateKernel(program, "initFunc", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create initFunc kernel");
      exit(1);
   }

   parScanFunction = clCreateKernel(program, "parScanCompose", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create parScanCompose kernel");
      exit(1);
   }

   parScanFunctionWithSubarrays = clCreateKernel(program, 
                                  "parScanComposeFromSubarrays", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create parScanComposeFromSubarrays kernel");
      exit(1);
   }

   parScanComposeFuncInc = clCreateKernel(program, "parScanComposeFuncInc", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create parScanComposeFuncInc kernel");
      exit(1);
   }

   findSeparators = clCreateKernel(program, "findSep", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create calcDel kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(initFunction, 0, sizeof(cl_mem), &input_buffer);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for initFunction:input buffer");
      exit(1);
   }
   err |= clSetKernelArg(initFunction, 1, sizeof(cl_int), &input_length);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for initFunction:input length");
      exit(1);
   }
   err |= clSetKernelArg(initFunction, 2, sizeof(cl_mem), &escape_buffer);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for initFunction:escape buffer");
      exit(1);
   }
   err |= clSetKernelArg(initFunction, 3, sizeof(cl_mem), &function_buffer);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for initFunction:function buffer");
      exit(1);
   }

   /*
   err = clSetKernelArg(parScanFunction, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(parScanFunction, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunction, 2, sizeof(cl_mem), &partial_buffer);
   err |= clSetKernelArg(parScanFunction, 3, sizeof(cl_int), &input_length);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for parScan");
      exit(1);
   }

   err = clSetKernelArg(parScanFunctionWithSubarrays, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 2, sizeof(cl_mem), &partial_buffer);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 3, sizeof(cl_int), &input_length);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for parScanWithSubarrays");
      exit(1);
   }*/

    err = clSetKernelArg(parScanComposeFuncInc, 0, sizeof(cl_mem), &function_buffer);
    err |= clSetKernelArg(parScanComposeFuncInc, 1, sizeof(cl_uint), &input_length);
    if(err != CL_SUCCESS){
       perror("Couldn't create a kernel argument for parScanComposeFuncInc");
       exit(1);
    }

   err = clSetKernelArg(findSeparators, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(findSeparators, 1, sizeof(cl_uint), &input_length);
   err |= clSetKernelArg(findSeparators, 2, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(findSeparators, 3, sizeof(cl_mem), &separator_buffer);
   err |= clSetKernelArg(findSeparators, 4, sizeof(cl_char), &firstCharacter);
   err |= clSetKernelArg(findSeparators, 5, sizeof(cl_mem), &output_buffer);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for findSeparators");
      exit(1);
   }

   /* Enqueue kernels */
   err = clEnqueueNDRangeKernel(queue, initFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the initFunc kernel");
      exit(1);
   }

   /*
   err = clEnqueueNDRangeKernel(queue, parScanFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the parScan kernel");
      exit(1);
   }

   err = clEnqueueNDRangeKernel(queue, parScanFunctionWithSubarrays, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the parScanWithSubarrays kernel");
      exit(1);
   }
   */

   
   err = clEnqueueNDRangeKernel(queue, parScanComposeFuncInc, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the parScanWithSubarrays");
      exit(1);
   }
   

   err = clEnqueueNDRangeKernel(queue, findSeparators, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the calcDel kernel");
      exit(1);
   }

   /* To ensure that parsing is done before reading the result*/
   clFinish(queue);

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), finalResults, 0, NULL, NULL);
   if(err != CL_SUCCESS) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   printf("[%d", finalResults[0]);
   for(int i=1; i<input_length; ++i) {
      printf(", %d", finalResults[i]);
   }
   printf("]\n");
   
   /* Deallocate resources */
   free(input_string);
   free(input_length);
   //free(local_array);
   free(finalResults);

   clReleaseDevice(device);

   clReleaseKernel(initFunction);
   clReleaseKernel(parScanFunction);
   clReleaseKernel(parScanFunctionWithSubarrays);
   clReleaseKernel(parScanComposeFuncInc);
   clReleaseKernel(findSeparators);

   clReleaseMemObject(input_buffer);
   clReleaseMemObject(function_buffer);
   clReleaseMemObject(output_buffer);
   clReleaseMemObject(escape_buffer);
   clReleaseMemObject(partial_buffer);
   clReleaseMemObject(separator_buffer);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
