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


void error_handler(cl_int err, char* message) {
   if(err == CL_SUCCESS)
      return;
   char* error_message = "";
   switch(err){
    // run-time and JIT compiler errors
    case 0: error_message = "CL_SUCCESS";
    case -1: error_message = "CL_DEVICE_NOT_FOUND";
    case -2: error_message = "CL_DEVICE_NOT_AVAILABLE";
    case -3: error_message = "CL_COMPILER_NOT_AVAILABLE";
    case -4: error_message = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: error_message = "CL_OUT_OF_RESOURCES";
    case -6: error_message = "CL_OUT_OF_HOST_MEMORY";
    case -7: error_message = "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: error_message = "CL_MEM_COPY_OVERLAP";
    case -9: error_message = "CL_IMAGE_FORMAT_MISMATCH";
    case -10: error_message = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: error_message = "CL_BUILD_PROGRAM_FAILURE";
    case -12: error_message = "CL_MAP_FAILURE";
    case -13: error_message = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: error_message = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: error_message = "CL_COMPILE_PROGRAM_FAILURE";
    case -16: error_message = "CL_LINKER_NOT_AVAILABLE";
    case -17: error_message = "CL_LINK_PROGRAM_FAILURE";
    case -18: error_message = "CL_DEVICE_PARTITION_FAILED";
    case -19: error_message = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: error_message = "CL_INVALID_VALUE";
    case -31: error_message = "CL_INVALID_DEVICE_TYPE";
    case -32: error_message = "CL_INVALID_PLATFORM";
    case -33: error_message = "CL_INVALID_DEVICE";
    case -34: error_message = "CL_INVALID_CONTEXT";
    case -35: error_message = "CL_INVALID_QUEUE_PROPERTIES";
    case -36: error_message = "CL_INVALID_COMMAND_QUEUE";
    case -37: error_message = "CL_INVALID_HOST_PTR";
    case -38: error_message = "CL_INVALID_MEM_OBJECT";
    case -39: error_message = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: error_message = "CL_INVALID_IMAGE_SIZE";
    case -41: error_message = "CL_INVALID_SAMPLER";
    case -42: error_message = "CL_INVALID_BINARY";
    case -43: error_message = "CL_INVALID_BUILD_OPTIONS";
    case -44: error_message = "CL_INVALID_PROGRAM";
    case -45: error_message = "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: error_message = "CL_INVALID_KERNEL_NAME";
    case -47: error_message = "CL_INVALID_KERNEL_DEFINITION";
    case -48: error_message = "CL_INVALID_KERNEL";
    case -49: error_message = "CL_INVALID_ARG_INDEX";
    case -50: error_message = "CL_INVALID_ARG_VALUE";
    case -51: error_message = "CL_INVALID_ARG_SIZE";
    case -52: error_message = "CL_INVALID_KERNEL_ARGS";
    case -53: error_message = "CL_INVALID_WORK_DIMENSION";
    case -54: error_message = "CL_INVALID_WORK_GROUP_SIZE";
    case -55: error_message = "CL_INVALID_WORK_ITEM_SIZE";
    case -56: error_message = "CL_INVALID_GLOBAL_OFFSET";
    case -57: error_message = "CL_INVALID_EVENT_WAIT_LIST";
    case -58: error_message = "CL_INVALID_EVENT";
    case -59: error_message = "CL_INVALID_OPERATION";
    case -60: error_message = "CL_INVALID_GL_OBJECT";
    case -61: error_message = "CL_INVALID_BUFFER_SIZE";
    case -62: error_message = "CL_INVALID_MIP_LEVEL";
    case -63: error_message = "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: error_message = "CL_INVALID_PROPERTY";
    case -65: error_message = "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: error_message = "CL_INVALID_COMPILER_OPTIONS";
    case -67: error_message = "CL_INVALID_LINKER_OPTIONS";
    case -68: error_message = "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: error_message = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: error_message = "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: error_message = "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: error_message = "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: error_message = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: error_message = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: error_message = "Unknown OpenCL error";
    }

   perror(error_message);
   if(message!=NULL)
      perror(message);
   exit(1);
   
}
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
   error_handler(err, "Couldn't create a context");

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

   escape_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                   input_length * sizeof(cl_char), NULL, &err);

   partial_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    num_groups * sizeof(cl_uint), NULL, &err);

   separator_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                      input_length * sizeof(cl_uint), NULL, &err);
   error_handler(err, "Couldn't create buffers");

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   error_handler(err, "Couldn't create a command queue");

   /* Creating kernels */
   initFunction = clCreateKernel(program, "initFunc", &err);
   error_handler(err, "Couldn't create initFunc kernel");

   parScanFunction = clCreateKernel(program, "parScanCompose", &err);
   error_handler(err, "Couldn't create parScanCompose kernel");

   parScanFunctionWithSubarrays = clCreateKernel(program, 
                                  "parScanComposeFromSubarrays", &err);
   error_handler(err, "Couldn't create parScanComposeFromSubarrays kernel");

   parScanComposeFuncInc = clCreateKernel(program, "parScanComposeFuncInc", &err);
   error_handler(err, "Couldn't create parScanComposeFuncInc kernel");

   findSeparators = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Couldn't create findSeparators kernel");

   /* Create kernel arguments */
   err = clSetKernelArg(initFunction, 0, sizeof(cl_mem), &input_buffer);
   error_handler(err, "Couldn't create a kernel argument for initFunction:input buffer");

   err |= clSetKernelArg(initFunction, 1, sizeof(cl_int), &input_length);
   error_handler(err, "Couldn't create a kernel argument for initFunction:input length");

   err |= clSetKernelArg(initFunction, 2, sizeof(cl_mem), &escape_buffer);
   error_handler(err, "Couldn't create a kernel argument for initFunction:escape buffer");

   err |= clSetKernelArg(initFunction, 3, sizeof(cl_mem), &function_buffer);
   error_handler(err, "Couldn't create a kernel argument for initFunction:function buffer");

   /*
   err = clSetKernelArg(parScanFunction, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(parScanFunction, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunction, 2, sizeof(cl_mem), &partial_buffer);
   err |= clSetKernelArg(parScanFunction, 3, sizeof(cl_int), &input_length);
   error_handler(err, "Couldn't create a kernel argument for parScan");

   err = clSetKernelArg(parScanFunctionWithSubarrays, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 1, NULL, &local_array);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 2, sizeof(cl_mem), &partial_buffer);
   err |= clSetKernelArg(parScanFunctionWithSubarrays, 3, sizeof(cl_int), &input_length);
   error_handler(err, "Couldn't create a kernel argument for parScanWithSubarrays");
   */

    err = clSetKernelArg(parScanComposeFuncInc, 0, sizeof(cl_mem), &function_buffer);
    err |= clSetKernelArg(parScanComposeFuncInc, 1, sizeof(cl_uint), &input_length);
    error_handler(err, "Couldn't create a kernel argument for parScanComposeFuncInc");

   err = clSetKernelArg(findSeparators, 0, sizeof(cl_mem), &function_buffer);
   err |= clSetKernelArg(findSeparators, 1, sizeof(cl_uint), &input_length);
   err |= clSetKernelArg(findSeparators, 2, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(findSeparators, 3, sizeof(cl_mem), &separator_buffer);
   err |= clSetKernelArg(findSeparators, 4, sizeof(cl_char), &firstCharacter);
   err |= clSetKernelArg(findSeparators, 5, sizeof(cl_mem), &output_buffer);
   error_handler(err, "Couldn't create a kernel argument for findSeparators");


   /* Enqueue kernels */
   err = clEnqueueNDRangeKernel(queue, initFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the initFunc kernel");


   /*
   err = clEnqueueNDRangeKernel(queue, parScanFunction, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the parScan kernel");

   err = clEnqueueNDRangeKernel(queue, parScanFunctionWithSubarrays, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the parScanWithSubarrays kernel");
   */

   
   err = clEnqueueNDRangeKernel(queue, parScanComposeFuncInc, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the parScanWithSubarrays");
   

   err = clEnqueueNDRangeKernel(queue, findSeparators, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   error_handler(err, "Couldn't enqueue the calcDel kernel");

   /* To ensure that parsing is done before reading the result*/
   clFinish(queue);

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), finalResults, 0, NULL, NULL);
   error_handler(err, "Couldn't read the buffer");

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
