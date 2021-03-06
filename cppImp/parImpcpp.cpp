
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "error_handler.hpp"
#include "helper_functions.hpp"

#define KERNEL_FILE "findSepNew.cl"
#define INPUT_FILE "Porto_taxi_data_test_partial_trajectories_orig.txt"
#define GLOBAL_SIZE 2048
#define LOCAL_SIZE 128

using namespace std;

int main(int argc, char** argv){

   string ifile = INPUT_FILE;
   if(argc==2) {
      ifile = argv[1];
   }

   //Get input file
   std::string chunk, residual;
   std::ifstream inputFile(ifile);

   if(!inputFile.is_open()) {
      exit(1);
   }


   //Throw away first line (has heading)
   std::string garbage;
   std::getline(inputFile, garbage);


   //read in chunk of data
   read_chunk(inputFile, chunk, residual);

   //Converting chunk to c-string and getting size
   cl_char* c_chunk = (cl_char*)(chunk.c_str());
   cl_uint chunkSize = chunk.size();

   //Setting global and local size; Global to next power of 2 from chunkSize
   size_t global_size = pad_num(chunkSize);
   size_t local_size = (LOCAL_SIZE <= global_size) ? LOCAL_SIZE : global_size;

   //For handling errors in OpenCL
   cl_int err;
   vector<cl_int> errors;


   //Create device, context, program, and queue
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   error_handler(err, "Couldn't create a context");

   cl_program program = build_program(context, device, KERNEL_FILE);

   cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
   error_handler(err, "Failed to create command queue");


   /** Creating buffers **/

   //buffer for data chunk from file
   cl_mem inputString = clCreateBuffer(context, CL_MEM_READ_ONLY |
            CL_MEM_COPY_HOST_PTR, chunkSize, c_chunk, &err);
   error_handler(err, "Failed to create 'inputString' buffer");

   //buffer for tracking position of '\n' characters
   cl_mem newLineBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_uint)*chunkSize, NULL, &err);
   error_handler(err, "Failed to create 'newLineBuff' buffer");

   //buffer to store the positions of valid separators
   cl_mem finalRes = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            sizeof(cl_uint)*chunkSize, NULL, &err);
   error_handler(err, "Failed to create 'finalRes' buffer");


   /** Creating kernels **/

   //marks locations of '\n' chars in a given buffer
   cl_kernel newLineAlt = clCreateKernel(program, "newLineAlt", &err);
   error_handler(err, "Failed to create 'newLineAlt' kernel");

   //compiles an array of the starts and ends of lines from newline buffer mentioned
   cl_kernel getLinePos = clCreateKernel(program, "getLinePos", &err);
   error_handler(err, "Failed to create 'getLinePos' kernel");
   
   //for performing the scan step of a parallel scanning addition on a global scale
   cl_kernel addScanStep = clCreateKernel(program, "addScanStep", &err);
   error_handler(err, "Failed to create 'addScanStep' kernel");
   
   //for performing the post-scan step of a parallel scanning addition on a global scale
   cl_kernel addPostScanStep = clCreateKernel(program, "addPostScanStep", &err);
   error_handler(err, "Failed to create 'addPostScanStep' kernel");
   
   //finds valid separators by parsing for delimited zones and returns positions of
   //separators not within those zones
   cl_kernel findSep = clCreateKernel(program, "findSep", &err);
   error_handler(err, "Failed to create 'findSep' kernel");

   //flips the order of the coordinates in the coordinate pairs of the polyline
   //for a given line ( TODO: WRTIE WHY BROKEN )
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
   

   /**
      Because there is no way to guarentee global synchronization between proceessing
      elements for each iterations of parallel scanning addition on the device, you must
      go back to the device as a means of sychronization at the end of each iteration.
      "addScanStep" and "addPostScanStep" run one iteration of the scan and post scan
      steps respectively.
   */

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
   size_t posSize = 2 * numLines;      //size of the buffer for the starts and ends of lines

   free(newLineArr);

   /**NOTE: Since you can specify the offset at which to read in the buffer and how much
            memory to read, you could just read the last value of the buffer (the only one 
            needed to find numLines) instead of the whole buffer to an array.
   */

   //Initalizing the array of positions for the starts and ends of lines in the array
   cl_uint * pos = (cl_uint *)calloc(posSize, sizeof(cl_uint));
   pos[0] = 0;                         //already know the start of the first line to be at 0
   pos[posSize-1] = chunkSize;         //already know the end of the last line to be at the
                                       //end of the chunk

   //Creating buffer for line start/end positions
   cl_mem posBuff = clCreateBuffer(context, CL_MEM_READ_WRITE 
            | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint)*posSize, pos, &err);
   error_handler(err, "Failed to create 'posBuff' buffer");

   /**
      Each line will have some number of valid separators. Thus the size of the array of
      positions for those valid separators could vary in size. We create a buffer to hold
      the number of valid separators for each line here.
   */
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
      cl_uint currSize = sizes[i/2];
      cout<<currStart<<": ";
      for(size_t j=0; j<currSize; ++j){
         cout << commPos[currStart + j] << " ";
      }
      cout << endl;
   }
   cout<<endl<<endl;
   

   for(size_t i=0; i<posSize; i+=2) {
      cl_uint currStart = pos[i]+7;
      if(sizes[i/2] == 0) continue;
      cl_uint currSize = sizes[i/2]-7; //7 irrelevant commas
      cl_uint finalSize = pos[i+1] - commPos[currStart] - 1;

      cl_uint posTracker2 = 0;
      cl_mem pos_ptr2 = clCreateBuffer(context, CL_MEM_READ_WRITE | 
            CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &posTracker2, &err);
      error_handler(err, "Failed to create 'pos_ptr' buffer");
      
      cl_mem output_line = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                           finalSize*sizeof(cl_char), NULL, &err);
      error_handler(err, "Failed to create 'output_line' buffer");

      errors.push_back(clSetKernelArg(flipCoords, 0, sizeof(cl_mem), &inputString));   //input_string
      errors.push_back(clSetKernelArg(flipCoords, 1, sizeof(cl_mem), &finalRes));      //start_positions
      errors.push_back(clSetKernelArg(flipCoords, 2, sizeof(cl_mem), &pos_ptr2));      //pos_ptr
      errors.push_back(clSetKernelArg(flipCoords, 3, sizeof(cl_mem), &output_line));   //output_string
      errors.push_back(clSetKernelArg(flipCoords, 4, sizeof(cl_uint), &currSize));     //num_pairs
      errors.push_back(clSetKernelArg(flipCoords, 5, sizeof(cl_uint), &finalSize));    //finalSize
      errors.push_back(clSetKernelArg(flipCoords, 6, sizeof(cl_uint), &currStart));   //currStart
      error_handler(errors, "Couldn't set args for flipCoords");

      err = clEnqueueNDRangeKernel(queue, flipCoords, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
      error_handler(err, "Couldn't enqueue flipCoords");

      cl_char* output_str = (cl_char*)malloc((finalSize+1)*sizeof(cl_char));
      err = clEnqueueReadBuffer(queue, output_line, CL_TRUE, 0, 
                  finalSize*sizeof(cl_char), output_str, 0, NULL, NULL);
      clFinish(queue);
      output_str[finalSize] = '\0';
      
      //cl_uint tag_length = commPos[currStart]-currStart+1;
      //cout<<chunk.substr(currStart, tag_length)<<'\"';
      for(cl_uint i=0; i<finalSize; ++i) {
         cout<<output_str[i];
      }
      cout << "\n" << endl;
      
      free(output_str);
      clReleaseMemObject(pos_ptr2);
      clReleaseMemObject(output_line);
   }




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
   clReleaseKernel(flipCoords);
   
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseDevice(device);
   clReleaseContext(context);
   
   free(sizes);
   free(pos);
   free(commPos);

   return 0;
}