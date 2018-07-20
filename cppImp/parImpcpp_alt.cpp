
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>

#include "error_handler.hpp"
#include "helper_functions.hpp"

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define KERNEL_FILE "findSepNew.cl"
#define INPUT_FILE "input.txt"
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

   std::string ifile = INPUT_FILE;
   if(argc==2) {
      ifile = argv[1];
   }

   //Get input file
   std::string chunk, residual;
   std::ifstream inputFile(ifile);
   if(!inputFile.is_open()) {
      exit(1);
   }
   std::string garbage;
   std::getline(inputFile, garbage);
   read_chunk_pp(inputFile, chunk, residual);

   cl_char* c_chunk = (cl_char*)(chunk.c_str());
   cl_uint chunkSize = chunk.size();
   // cl_char* c_chunk = (cl_char*)malloc(chunkSize);
   // for(int i=0; i<chunkSize; ++i){
   //    c_chunk[i] = chunk[i];
   // }

   size_t global_size = pad_num(chunkSize);
   size_t local_size = (LOCAL_SIZE <= global_size) ? LOCAL_SIZE : global_size;

   cl_int err;
   vector<cl_int> errors;


   //Create device, context, program, and queue
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   cl_program program = build_program(context, device, KERNEL_FILE);

   cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
   error_handler(err, "Failed to create command queue");


   //Create buffers
   cl_mem inputString = clCreateBuffer(context, CL_MEM_READ_ONLY |
            CL_MEM_COPY_HOST_PTR, chunkSize, c_chunk, &err);
   error_handler(err, "Failed to create 'inputString' buffer");

   cl_mem newLineBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_uint)*chunkSize, NULL, &err);
   error_handler(err, "Failed to create 'newLineBuff' buffer");

   cl_mem finalRes = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_uint)*chunkSize, NULL, &err);
   error_handler(err, "Failed to create 'finalRes' buffer");


   //Creating kernels
   cl_kernel newLineAlt = clCreateKernel(program, "newLineAlt", &err);
   error_handler(err, "Failed to create 'newLineAlt' kernel");

   cl_kernel getLinePos = clCreateKernel(program, "getLinePos", &err);
   error_handler(err, "Failed to create 'getLinePos' kernel");
   
   cl_kernel addScanStep = clCreateKernel(program, "addScanStep", &err);
   error_handler(err, "Failed to create 'addScanStep' kernel");
   
   cl_kernel addPostScanStep = clCreateKernel(program, "addPostScanStep", &err);
   error_handler(err, "Failed to create 'addPostScanStep' kernel");
   
   cl_kernel findSep = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Failed to create 'findSep' kernel");

   cl_kernel flipCoords = clCreateKernel(program, "flipCoords", &err);
   error_handler(err, "Failed to create 'flipCoords' kernel");


   //Running newLineAlt
   errors.push_back(clSetKernelArg(newLineAlt, 0, sizeof(cl_mem), &inputString));
   errors.push_back(clSetKernelArg(newLineAlt, 1, sizeof(cl_mem), &newLineBuff));
   errors.push_back(clSetKernelArg(newLineAlt, 2, sizeof(cl_uint), &chunkSize));
   error_handler(errors, "Failed to set a kernel arguement for 'newLineAlt'");

   err = clEnqueueNDRangeKernel(queue, newLineAlt, 1, NULL, 
            &global_size, &local_size, 0, NULL, NULL);
   error_handler(err, "Failed to enqueue 'newLineAlt' kernel");
   

   //Running addScanStep
   cl_uint depth = lg(global_size);
   for(cl_uint d=0; d < depth; ++d){
      errors.push_back(clSetKernelArg(addScanStep, 0, sizeof(cl_mem), &newLineBuff));
      errors.push_back(clSetKernelArg(addScanStep, 1, sizeof(cl_uint), &chunkSize));
      errors.push_back(clSetKernelArg(addScanStep, 2, sizeof(cl_uint), &d));
      error_handler(errors, "Failed to set a kernel arguement for 'addScanStep'");

      err = clEnqueueNDRangeKernel(queue, addScanStep, 1, NULL, 
               &global_size, &local_size, 0, NULL, NULL);
      error_handler(err, "Failed to enqueue 'addScanStep' kernel");
   }

   //Running addPostScanStep
   for(cl_uint stride = global_size/4; stride > 0; stride /= 2){
      errors.push_back(clSetKernelArg(addPostScanStep, 0, sizeof(cl_mem), &newLineBuff));
      errors.push_back(clSetKernelArg(addPostScanStep, 1, sizeof(cl_uint), &chunkSize));
      errors.push_back(clSetKernelArg(addPostScanStep, 2, sizeof(cl_uint), &stride));
      error_handler(errors, "Failed to set a kernel arguement for 'addPostScanStep'");

      err = clEnqueueNDRangeKernel(queue, addPostScanStep, 1, NULL, 
               &global_size, &local_size, 0, NULL, NULL);
      error_handler(err, "Failed to enqueue 'addPostScanStep' kernel");
   }

   clFinish(queue);

   //Setting up buffers for line positions and final result sizes
   cl_uint * newLineArr = (cl_uint *)malloc(sizeof(cl_uint)*chunkSize);
   err = clEnqueueReadBuffer(queue, newLineBuff, CL_TRUE, 0, 
            sizeof(cl_uint)*chunkSize, newLineArr, 0, NULL ,NULL);
   error_handler(err, "Failed to read 'newLineBuff' buffer");

   size_t numLines = newLineArr[chunkSize-1] + 1;
   size_t posSize = 2 * numLines;

   free(newLineArr);
   
   cl_uint * pos = (cl_uint *)calloc(posSize, sizeof(cl_uint));
   pos[0] = 0;
   pos[posSize-1] = chunkSize;

   cl_mem posBuff = clCreateBuffer(context, CL_MEM_READ_WRITE 
            | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint)*posSize, pos, &err);
   error_handler(err, "Failed to create 'posBuff' buffer");

   cl_mem resSizes = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_uint)*numLines, NULL, &err);
   error_handler(err, "Failed to create 'resSizes' buffer");


   //Running getLinePos
   errors.push_back(clSetKernelArg(getLinePos, 0, sizeof(cl_mem), &newLineBuff));
   errors.push_back(clSetKernelArg(getLinePos, 1, sizeof(cl_mem), &posBuff));
   errors.push_back(clSetKernelArg(getLinePos, 2, sizeof(cl_uint), &chunkSize));
   error_handler(errors, "Failed to set a kernel arguement for 'getLinePos'");

   err = clEnqueueNDRangeKernel(queue, getLinePos, 1, NULL, 
            &global_size, &local_size, 0, NULL, NULL);
   error_handler(err, "Failed to enqueue 'getLinePos' kernel");

   cl_uint posTracker = 0;
   cl_mem pos_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE | 
            CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &posTracker, &err);
   error_handler(err, "Failed to create 'pos_ptr' buffer");

   //Running findSep
   errors.push_back(clSetKernelArg(findSep, 0, sizeof(cl_mem), &inputString));      //input_string
   errors.push_back(clSetKernelArg(findSep, 1, sizeof(cl_mem), &posBuff));          //input_pos
   errors.push_back(clSetKernelArg(findSep, 2, sizeof(cl_mem), &pos_ptr));          //pos_ptr
   errors.push_back(clSetKernelArg(findSep, 3, sizeof(cl_uint)*chunkSize, NULL));   //separators
   errors.push_back(clSetKernelArg(findSep, 4, sizeof(cl_mem), &finalRes));         //finalResults
   errors.push_back(clSetKernelArg(findSep, 5, sizeof(cl_mem), &resSizes));         //result_sizes
   errors.push_back(clSetKernelArg(findSep, 6, sizeof(cl_char)*local_size, NULL));  //lstring
   errors.push_back(clSetKernelArg(findSep, 7, sizeof(cl_char)*local_size, NULL));  //escape
   errors.push_back(clSetKernelArg(findSep, 8, sizeof(cl_char)*local_size, NULL));  //function
   errors.push_back(clSetKernelArg(findSep, 9, sizeof(cl_uint), &numLines));        //lines
   errors.push_back(clSetKernelArg(findSep, 10, sizeof(cl_uint), NULL));            //len
   errors.push_back(clSetKernelArg(findSep, 11, sizeof(cl_uint), NULL));            //curr_pos
   errors.push_back(clSetKernelArg(findSep, 12, sizeof(cl_uint), NULL));            //prev_escape
   errors.push_back(clSetKernelArg(findSep, 13, sizeof(cl_char), NULL));            //prev_function
   errors.push_back(clSetKernelArg(findSep, 14, sizeof(cl_uint), NULL));            //prev_sep
   errors.push_back(clSetKernelArg(findSep, 15, sizeof(cl_uint), NULL));            //elems_scanned
   errors.push_back(clSetKernelArg(findSep, 16, sizeof(cl_char), NULL));            //first_char
   error_handler(errors, "Failed to set a kernel arguement for 'findSep'");

   err = clEnqueueNDRangeKernel(queue, findSep, 1, NULL, 
            &global_size, &local_size, 0, NULL, NULL);
   error_handler(err, "Failed to enqueue 'findSep' kernel");

   
   //Reading from results buffers
   cl_uint * commPos = (cl_uint *)malloc(sizeof(cl_uint)*chunkSize);
   cl_uint * sizes = (cl_uint *)malloc(sizeof(cl_uint)*numLines);

   err = clEnqueueReadBuffer(queue, finalRes, CL_TRUE, 0, 
            sizeof(cl_uint)*chunkSize, commPos, 0, NULL, NULL);
   error_handler(err, "Failed to read 'finalRes' buffer");
   
   err = clEnqueueReadBuffer(queue, resSizes, CL_TRUE, 0, 
            sizeof(cl_uint)*numLines, sizes, 0, NULL, NULL);
   error_handler(err, "Failed to read 'resSizes' buffer");

   err = clEnqueueReadBuffer(queue, posBuff, CL_TRUE, 0, 
            sizeof(cl_uint)*posSize, pos, 0, NULL, NULL);
   error_handler(err, "Failed to read 'posBuff' buffer");

   // Printing out results
   for(size_t i=0; i<posSize; i+=2){
      int currStart = pos[i];
      int currSize = sizes[i/2];
      cout<<currStart<<": ";
      for(size_t j=0; j<currSize; ++j){
         cout << commPos[currStart + j] << " ";
      }
      cout << endl;
   }

   for(size_t i=0; i<posSize; i+=2) {
      cl_int currStart = pos[i];
      cl_uint currSize = sizes[i/2]-7;//7 irrelevant commas
      cl_uint finalSize = (pos[i+1]-commPos[currStart+7]-1);
      cl_uint* currCommas = &commPos[currStart+7];
      currCommas[0]+=2;
      cl_uint posTracker2 = 0;
      cl_mem pos_ptr2 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
            CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &posTracker2, &err);
      error_handler(err, "Failed to create 'pos_ptr' buffer");
      cl_mem startPos = clCreateBuffer(context, CL_MEM_READ_WRITE | 
            CL_MEM_COPY_HOST_PTR, currSize*sizeof(cl_uint), &currCommas, &err);
      error_handler(err, "Failed to create 'startPos' buffer");
      cl_mem currSizeBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | 
            CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &currSize, &err);
      error_handler(err, "Failed to create 'startPos' buffer");
      cl_mem output_line = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                           finalSize*sizeof(cl_char), NULL, &err);
      error_handler(err, "Failed to create 'output_line' buffer");

      errors.push_back(clSetKernelArg(flipCoords, 0, sizeof(cl_mem), &inputString));   //input_string
      errors.push_back(clSetKernelArg(flipCoords, 1, sizeof(cl_mem), &startPos));      //start_positions
      errors.push_back(clSetKernelArg(flipCoords, 2, sizeof(cl_mem), &currSizeBuffer));     //num_pairs
      errors.push_back(clSetKernelArg(flipCoords, 3, sizeof(cl_mem), &pos_ptr2));      //pos_ptr
      errors.push_back(clSetKernelArg(flipCoords, 4, sizeof(cl_uint), &finalSize));    //finalSize
      errors.push_back(clSetKernelArg(flipCoords, 5, sizeof(cl_uint), NULL));          //curr_pos
      errors.push_back(clSetKernelArg(flipCoords, 6, sizeof(cl_uint), NULL));          //loc_length
      errors.push_back(clSetKernelArg(flipCoords, 7, sizeof(cl_uint), NULL));          //mid
      errors.push_back(clSetKernelArg(flipCoords, 8, sizeof(cl_uint), NULL));          //y_len
      errors.push_back(clSetKernelArg(flipCoords, 9, sizeof(cl_mem), &output_line));   //output_string
      error_handler(errors, "Couldn't set args for flipCoords");

      err = clEnqueueNDRangeKernel(queue, flipCoords, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      error_handler(err, "Couldn't enqueue flipCoords");
      clFinish(queue);

      cl_char* output_str = (cl_char*)malloc(finalSize*sizeof(cl_char));
      err = clEnqueueReadBuffer(queue, output_line, CL_TRUE, 0, 
                  finalSize*sizeof(cl_char), output_str, 0, NULL, NULL);

      cl_uint tag_length = commPos[currStart]-currStart+1;
      cout<<chunk.substr(currStart, tag_length);
      for(cl_uint i=0; i<finalSize; ++i) {
         cout<<output_str[i];
      }
      cout<<endl<<endl;

      free(output_str);
      clReleaseMemObject(pos_ptr2);
      clReleaseMemObject(startPos);
      clReleaseMemObject(currSizeBuffer);
      clReleaseMemObject(output_line);
   }

   free(commPos);
   free(sizes);
   free(pos);

   //Freeing CL Objects
   clReleaseMemObject(inputString);
   clReleaseMemObject(newLineBuff);
   clReleaseMemObject(posBuff);
   clReleaseMemObject(finalRes);
   clReleaseMemObject(resSizes);
   clReleaseMemObject(pos_ptr);

   clReleaseKernel(newLineAlt);
   clReleaseKernel(getLinePos);
   clReleaseKernel(addScanStep);
   clReleaseKernel(addPostScanStep);
   clReleaseKernel(findSep);

   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseDevice(device);
   clReleaseContext(context);

   return 0;
}