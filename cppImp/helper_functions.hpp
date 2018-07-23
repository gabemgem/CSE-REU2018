#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <CL/cl.hpp>

#define CHUNK_SIZE 2048

#ifndef DEVICE_TYPE
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

/* Reads in a chunk of data from file. Ensures that  the chunk
   starts/ends on with a complete line.
*/
void read_chunk(std::ifstream & file, std::string & chunk, std::string & residual){
   unsigned int size = chunk.size();
   std::string line;
   while(std::getline(file, line)){
      if(size + line.size() > CHUNK_SIZE){
         residual = line + '\n';
         return;
      }

      chunk += line + '\n';
      line.clear();
      size = chunk.size();
   }
}

/* Reads in a chunk of data from file. Ensures that  the chunk
   starts/ends on with a complete line. Stores beginning and 
   end locations of lines.
*/
void read_chunk_with_nums(std::ifstream &file, std::string &chunk, 
                          std::string &residual, unsigned int* linenums){
   unsigned int size = chunk.size();
   unsigned int loc = 0, loccount = 0;
   std::string line;
   while(std::getline(file, line)){
      if(size + line.size() > CHUNK_SIZE){
         //removing ending newline
         chunk = chunk.substr(0, size-1);
         residual = line;
         return;
      }
      line+="\n";
      linenums[2*loccount] = loc;
      loc+=line.size();
      linenums[2*loccount+1] = loc;
      loc++;
      loccount++;
      chunk += line;
      size = chunk.size();
   }
}

/* Returns the next power of 2 from old */
cl_int pad_num(cl_int old) {
   cl_int new_val = 1;
   while(old > new_val) {
      new_val <<= 1;
   }
   return new_val;
}

cl_uint lg(cl_int val){
   cl_uint out = 0, cpy = val;
   while(cpy > 1){
      cpy >>= 1;
      ++out;
   }
   if((1 << out) < val) ++out;

   return out;
}

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
}

#endif /* helper_functions.h */