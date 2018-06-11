#define INPUT_FILE "input.txt"
#define PROGRAM_FILE "findSep.cl"
#define KERNEL_FUNC "findSep"
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

   fp = fopen(INPUT_FILE, 'r');
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

int main() {

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t local_size, global_size;

   /* Data and buffers */
   char* input_string;
   cl_int* input_length;
   char* specChars = ",[]\\";
   size_t specChars_length = strlen(specChars);

   cl_mem input_buffer, output_buffer;
   cl_int num_groups;

   /* Get data from file */
   //call to get line requires free at end
   read_from_file(input_string, input_length);

   /*Shared memory for kernel*/
   cl_uint* out_array = malloc((*input_length) * sizeof(cl_uint));
   cl_uint* escape = malloc((*input_length) * sizeof(cl_uint));
   cl_uint* open = malloc((*input_length) * sizeof(cl_uint));
   cl_uint* close = malloc((*input_length) * sizeof(cl_uint));
   cl_uint** function = malloc((*input_length) * sizeof(cl_uint*));
   cl_uint* delimited = malloc((*input_length) * sizeof(cl_uint));
   cl_uint* separator = malloc((*input_length) * sizeof(cl_uint));

   /* Create device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create data buffer */
   global_size = 128;//Total NUM THREADS
   local_size = 8;//NUM THREADS per BLOCK
   num_groups = global_size/local_size;//NUM BLOCKS
   
   /* Create buffer for input string */
   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, (*input_length) * sizeof(char), input_string, &err);

   output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, (*input_length) * sizeof(cl_uint), out_array, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   /* Change according to your kernel */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
   err |= clSetKernelArg(kernel, 2, specChars_length, &specChars);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_int), input_length);
   err |= clSetKernelArg(kernel, 4, (*input_length) * sizeof(cl_uint), &escape);
   err |= clSetKernelArg(kernel, 5, sizeof(&function), &function);                  //not actual size
   err |= clSetKernelArg(kernel, 6, (*input_length) * sizeof(cl_uint), &delimited);
   err |= clSetKernelArg(kernel, 7, (*input_length) * sizeof(cl_uint), &separator);
   if(err != CL_SUCCESS) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   /* Enqueue kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err != CL_SUCCESS) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }
   
   /* To ensure that parsing is done before reading the result (not sure if needed)*/
   clFinish(queue);

   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
         (*input_length) * sizeof(cl_uint), out_array, 0, NULL, NULL);
   if(err != CL_SUCCESS) {
      perror("Couldn't read the buffer");
      exit(1);
   }
   
   /* Deallocate resources */
   free(input_string);
   free(input_length);
   free(out_array);
   free(escape);
   free(function);
   free(delimited);
   free(separator);
   clReleaseKernel(kernel);
   clReleaseMemObject(output_buffer);
   clReleaseMemObject(input_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
