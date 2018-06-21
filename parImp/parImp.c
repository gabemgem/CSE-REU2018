#define VERBOSE 0

#define INPUT_FILE "input.txt"
#define PROGRAM_FILE "findSep.cl"

#define NUM_LINES 2
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

cl_uint read_from_file(FILE* fp, char* line, cl_int guess, char* eof) {
   if(!fp) {
      printf("Couldn't open input file");
      exit(1);
   }

   cl_uint counter=0, size=guess;
   cl_char c;
   while((c = fgetc(fp)) != '\n') {
      if(feof(fp)) {
         eof[0]=1;
         break;
      }
      line[counter] = c;
      counter++;
      if(counter==size) {
         size*=2;
         line = (char*)realloc(line, size * sizeof(cl_char));
      }

   }
   if(counter<size) {
      for(int i = counter; i < size; ++i) {
         line[i] = ' ';
      }
   }
   printf("%s\n", line);
   
   return size;

}

/* Padding string w/ spaces to a length of next power of 2.
   Stores padded string and new length in parameters.
*/
void pad_string(char** str, cl_int* len){ 
   //probably a faster implementation
   cl_int new_len = 1;


   while(*len > new_len){
      if(*len==new_len) {
         return;
      }
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

cl_int pad_num(cl_int old) {
   cl_int new = 1;
   while(old>new) {
      new<<=1;
   }
   return new;
}

int main(int argc, char** argv) {


   cl_int err;


   cl_int guess = 16;
   if(argc==2) {
      guess = atoi(argv[1]) + 1;
      guess = pad_num(guess);
   }
   cl_int* input_length = malloc(NUM_LINES * sizeof(cl_int));
   char** input_string = malloc(NUM_LINES * sizeof(char*));
   char* eof = malloc(sizeof(char));
   *eof = 0;
   cl_int nlines = NUM_LINES;


<<<<<<< HEAD
   /* Create work group size */
   size_t global_size = 64;//Total NUM THREADS
   size_t local_size = 8;//NUM THREADS per BLOCK
   cl_int num_groups = global_size/local_size;//NUM BLOCKS
   
   /* Shared memory for parallel scan kernels */
   //cl_uint* local_array;
   /* Shared memory for findSep */
   cl_char firstCharacter = (input_string[0] == OPEN);
   cl_uint* finalResults = malloc(input_length * sizeof(cl_uint));
   
   /* Create buffers */
   cl_mem input_buffer, function_buffer, output_buffer;
   cl_mem escape_buffer, partial_buffer, separator_buffer;
=======
   FILE* fp = fopen(INPUT_FILE, "r");
   if(!fp) {
      printf("Couldn't open input file");
      exit(1);
   }
>>>>>>> Staging

   /* Get data from file */
   for(int i=0; i<NUM_LINES; ++i) {
      input_string[i] = malloc(guess*sizeof(char));
      input_length[i] = read_from_file(fp, input_string[i], guess, eof);
      printf("%s\n", input_string[i]);
      /* Pads string to length of next power of 2 with spaces */
      pad_string(&input_string[i], &input_length[i]);
      
      if(*eof!=0) {
         nlines = i+1;
         break;
      }
   }

   fclose(fp);


   if(VERBOSE){
      printf("%d\n", input_length[0]);
   }

   /* Create device and context */
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   /* Build program */
   cl_program program = build_program(context, device, PROGRAM_FILE);

   /* Create a command queue */
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
   error_handler(err, "Couldn't create a command queue");

   /* Creating kernels */
   cl_kernel initFunction = clCreateKernel(program, "initFunc", &err);
   error_handler(err, "Couldn't create initFunc kernel");

   cl_kernel scanStep = clCreateKernel(program, "scanStep", &err);
   error_handler(err, "Couldn't create scanStep kernel");

   cl_kernel postScanIncStep = clCreateKernel(program, "postScanIncStep", &err);
   error_handler(err, "Couldn't create postScanIncStep kernel");

   cl_kernel addScanStep = clCreateKernel(program, "addScanStep", &err);
   error_handler(err, "Couldn't create addScanStep kernel");

   cl_kernel addPostScanIncStep = clCreateKernel(program, "addPostScanIncStep", &err);
   error_handler(err, "Couldn't create addPostScanIncStep kernel");

   cl_kernel findSep = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Couldn't create findSep kernel");

   cl_kernel compressRes = clCreateKernel(program, "compressResults", &err);
   error_handler(err, "Couldn't create compressRes kernel");


   printf("%d\n", nlines);
   printf("%s\n", input_string[0]);
   printf("%s\n", input_string[1]);
   printf("%d\n", input_length[0]);
   printf("%d\n", input_length[1]);

   for(int l = 0; l < nlines; ++l) {
      /* Create work group size */
      size_t global_size = input_length[l];//Total NUM THREADS
      size_t local_size = 8;//NUM THREADS per BLOCK
      
      /* Shared memory for findSep */
      cl_char firstCharacter = (input_string[l][0] == SEP);
      cl_uint* finalResults = malloc(input_length[l] * sizeof(cl_uint));
      
      /* Create buffers */

      cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
            CL_MEM_COPY_HOST_PTR, input_length[l] * sizeof(char), input_string[l], &err);
      cl_mem function_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        input_length[l] * sizeof(cl_char), NULL, &err);
      cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                      input_length[l] * sizeof(cl_uint), NULL, &err);
      cl_mem escape_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                      input_length[l] * sizeof(cl_char), NULL, &err);

      error_handler(err, "Couldn't create buffers");


      // Setting up and running init function
      err = clSetKernelArg(initFunction, 0, sizeof(cl_mem), &input_buffer);
      err |= clSetKernelArg(initFunction, 1, sizeof(cl_int), &input_length[l]);
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
   

      cl_uint depth = lg(input_length[l]), d;
      for(d=0; d<depth; ++d){
         err = clSetKernelArg(scanStep, 0, sizeof(cl_mem), &function_buffer);
         err |= clSetKernelArg(scanStep, 1, sizeof(cl_uint), &input_length[l]);
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
      for(stride = input_length[l]/4; stride > 0; stride /= 2){
         err = clSetKernelArg(postScanIncStep, 0, sizeof(cl_mem), &function_buffer);
         err |= clSetKernelArg(postScanIncStep, 1, sizeof(cl_uint), &input_length[l]);
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
      err |= clSetKernelArg(findSep, 1, sizeof(cl_uint), &input_length[l]);
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
         err |= clSetKernelArg(addScanStep, 1, sizeof(cl_uint), &input_length[l]);
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

      for(stride = input_length[l]/4; stride > 0; stride /= 2){
         err = clSetKernelArg(addPostScanIncStep, 0, sizeof(cl_mem), &output_buffer);
         err |= clSetKernelArg(addPostScanIncStep, 1, sizeof(cl_uint), &input_length[l]);
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
            input_length[l] * sizeof(cl_uint), finalResults, 0, NULL, NULL);
      error_handler(err, "Couldn't read the buffer");

      
      printf("%s\n", input_string[l]);
      for(int i=0; i<input_length[l]; ++i) {
         printf("%d", finalResults[i]);
      }

      cl_uint num = finalResults[input_length[l]-1];
      printf("\n%d\n", num);
      cl_mem compressedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                      num * sizeof(cl_uint), NULL, &err);

      
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

      printf("\n");
      for(int i=0; i<num; ++i) {
         printf("%d, ", compressedResults[i]);
      }
      printf("\n");

      free(finalResults);
      free(compressedResults);
      clReleaseMemObject(input_buffer);
      clReleaseMemObject(function_buffer);
      clReleaseMemObject(output_buffer);
      clReleaseMemObject(escape_buffer);
      clReleaseMemObject(compressedBuffer);

   }

<<<<<<< HEAD
   /* Read the kernel's output */
   err = clEnqueueReadBuffer(queue, separator_buffer, CL_TRUE, 0,
         input_length * sizeof(cl_uint), finalResults, 0, NULL, NULL);
   error_handler(err, "Couldn't read the buffer");
=======
>>>>>>> Staging

   
   
   /* Deallocate resources */
   for(int j = 0; j<NUM_LINES; ++j) {
      free(input_string[j]);
   }
   free(input_string);
   free(input_length);
   free(eof);
   

   clReleaseDevice(device);

   clReleaseKernel(initFunction);
   clReleaseKernel(scanStep);
   clReleaseKernel(postScanIncStep);
   clReleaseKernel(findSep);


   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   if(VERBOSE){
      printf("Resources deallocated\n");
   }

   return 0;
}
