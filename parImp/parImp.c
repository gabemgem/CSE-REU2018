#define INPUT_FILE "input.txt"
#define PROGRAM_FILE "findSep.cl"
#define NUM_QUEUES 4
//#define INPUT_SIZE 64//Use if input size is already known

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
   cl_kernel calculateFunction;
   cl_kernel parScanFunction;
   cl_kernel parScanFunctionWithSubarrays;
   cl_kernel calculateDelimited;
   static cl_command_queue queue[NUM_QUEUES];
   cl_int i, j, err;
   size_t local_size, global_size;

   /* Data and buffers */
   char* input_string;
   cl_int input_length;
   char* specChars = ",[]\\";
   size_t specChars_length = strlen(specChars);

   cl_mem input_buffer, function_buffer;
   cl_int num_groups;

   /* Get data from file */
   //call to get line requires free at end
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
   global_size = 128;//Total NUM THREADS
   local_size = 8;//NUM THREADS per BLOCK
   num_groups = global_size/local_size;//NUM BLOCKS
   
   /* Shared memory for kernel */
   cl_uint* escape = malloc(input_length * sizeof(cl_uint));
   cl_uint* open = malloc(input_length * sizeof(cl_uint));
   cl_uint* close = malloc(input_length * sizeof(cl_uint));
   cl_uint* function = malloc(input_length * sizeof(cl_uint) * 2);
   cl_uint* delimited = malloc(input_length * sizeof(cl_uint));
   cl_uint* separator = malloc(input_length * sizeof(cl_uint));
   cl_uint* local_array;
   cl_uint* partial_results = malloc((global_size/local_size) * sizeof(cl_uint));

   /* Create buffer for input string */
   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, input_length * sizeof(char), input_string, &err);

   function_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, input_length * sizeof(cl_uint)*2, function, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create input and output buffers");
      exit(1);   
   };

   /* Create a command queues */
   for(i=0; i<NUM_QUEUES; ++i) {
      queue[i] = clCreateCommandQueue(context, device, 0, &err);
      if(err != CL_SUCCESS) {
         perror("Couldn't create a command queue");
         exit(1);   
      };
   }

   /* Creating kernels */
   calculateFunction = clCreateKernel(program, "calcFunc", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create calcFunc kernel");
      exit(1);
   };

   parScanFunction = clCreateKernel(program, "parScanCompose", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create parScanCompose kernel");
      exit(1);
   };

   parScanFunctionWithSubarrays = clCreateKernel(program, 
                                  "parScanComposeWithSubarrays", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create parScanComposeWithSubarrays kernel");
      exit(1);
   };

   calculateDelimited = clCreateKernel(program, "calcDel", &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create calcDel kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(calculateFunction, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(calculateFunction, 1, specChars_length, &specChars);
   err |= clSetKernelArg(calculateFunction, 2, sizeof(cl_int), &input_length);
   err |= clSetKernelArg(calculateFunction, 3, input_length * sizeof(cl_uint), &escape);
   err |= clSetKernelArg(calculateFunction, 4, sizeof(cl_mem), &function);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for calcFun");
      exit(1);
   }

   err = clSetKernelArg(parScanFunction, 0, sizeof(cl_mem), &function);
   err |= clSetKernelArg(parScanFunction, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunction, 2, (global_size/local_size) * sizeof(cl_uint), partial_results);
   err |= clSetKernelArg(parScanFunction, 3, sizeof(cl_int), *input_length);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for parScan");
      exit(1);
   }

   err = clSetKernelArg(parScanFunctionWithSubarrays, 0, sizeof(cl_mem), &function);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 2, (global_size/local_size) * sizeof(cl_uint), partial_results);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 3, sizeof(cl_int), *input_length);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument for parScanWithSubarrays");
      exit(1);
   }

   /* Enqueue kernel */
   err = clEnqueueNDRangeKernel(queue[0], calculateFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the calcFunc");
      exit(1);
   }

   err = clEnqueueNDRangeKernel(queue[1], parScanFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the parScan");
      exit(1);
   }

   err = clEnqueueNDRangeKernel(queue[2], parScanFunctionWithSubarrays, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the parScanWithSubarrays");
      exit(1);
   }

   err = clEnqueueNDRangeKernel(queue[3], calculateDelimited, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the calcDel");
      exit(1);
   }
<<<<<<< HEAD
<<<<<<< HEAD
   
   /* To ensure that parsing is done before reading the result (not sure if needed)*/
   clFinish(queue[0]);
   clFinish(queue[1]);
   clFinish(queue[2]);
   clFinish(queue[3]);

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue[0], function_buffer, CL_TRUE, 0,
         (*input_length) * sizeof(cl_uint), function, 0, NULL, NULL);
=======

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, function_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), function, 0, NULL, NULL);
>>>>>>> 1e8d2aa7c3033d3941ad2db0f17ed6d4318e5bde
=======

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, function_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), function, 0, NULL, NULL);
>>>>>>> 1e8d2aa7c3033d3941ad2db0f17ed6d4318e5bde
   if(err != CL_SUCCESS) {
      perror("Couldn't read the buffer");
      exit(1);
   }
   
   /* Deallocate resources */
   free(input_string);
   free(input_length);
   free(escape);
   free(function);
   free(delimited);
   free(separator);
   free(specChars);

   clReleaseKernel(calculateFunction);
   clReleaseKernel(parScanFunction);
   clReleaseKernel(parScanFunctionWithSubarrays);
   clReleaseKernel(calculateDelimited);

   clReleaseMemObject(function_buffer);
   clReleaseMemObject(input_buffer);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
