#ifndef ERR_HANDLER
#define ERR_HANDLER

#include <iostream>
#include <string>
#include <CL/cl.hpp>

std::string get_error_message(cl_int err){
   std::string error_message;
   
   switch(err){
      // run-time and JIT compiler errors
      case -1: error_message = "CL_DEVICE_NOT_FOUND";
               break;
      case -2: error_message = "CL_DEVICE_NOT_AVAILABLE";
               break;
      case -3: error_message = "CL_COMPILER_NOT_AVAILABLE";
               break;
      case -4: error_message = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
               break;
      case -5: error_message = "CL_OUT_OF_RESOURCES";
               break;
      case -6: error_message = "CL_OUT_OF_HOST_MEMORY";
               break;
      case -7: error_message = "CL_PROFILING_INFO_NOT_AVAILABLE";
               break;
      case -8: error_message = "CL_MEM_COPY_OVERLAP";
               break;
      case -9: error_message = "CL_IMAGE_FORMAT_MISMATCH";
               break;
      case -10: error_message = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
               break;
      case -11: error_message = "CL_BUILD_PROGRAM_FAILURE";
               break;
      case -12: error_message = "CL_MAP_FAILURE";
               break;
      case -13: error_message = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
               break;
      case -14: error_message = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
               break;
      case -15: error_message = "CL_COMPILE_PROGRAM_FAILURE";
               break;
      case -16: error_message = "CL_LINKER_NOT_AVAILABLE";
               break;
      case -17: error_message = "CL_LINK_PROGRAM_FAILURE";
               break;
      case -18: error_message = "CL_DEVICE_PARTITION_FAILED";
               break;
      case -19: error_message = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
               break;

      // compile-time errors
      case -30: error_message = "CL_INVALID_VALUE";
               break;
      case -31: error_message = "CL_INVALID_DEVICE_TYPE";
               break;
      case -32: error_message = "CL_INVALID_PLATFORM";
               break;
      case -33: error_message = "CL_INVALID_DEVICE";
               break;
      case -34: error_message = "CL_INVALID_CONTEXT";
               break;
      case -35: error_message = "CL_INVALID_QUEUE_PROPERTIES";
               break;
      case -36: error_message = "CL_INVALID_COMMAND_QUEUE";
               break;
      case -37: error_message = "CL_INVALID_HOST_PTR";
               break;
      case -38: error_message = "CL_INVALID_MEM_OBJECT";
               break;
      case -39: error_message = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
               break;
      case -40: error_message = "CL_INVALID_IMAGE_SIZE";
               break;
      case -41: error_message = "CL_INVALID_SAMPLER";
               break;
      case -42: error_message = "CL_INVALID_BINARY";
               break;
      case -43: error_message = "CL_INVALID_BUILD_OPTIONS";
               break;
      case -44: error_message = "CL_INVALID_PROGRAM";
               break;
      case -45: error_message = "CL_INVALID_PROGRAM_EXECUTABLE";
               break;
      case -46: error_message = "CL_INVALID_KERNEL_NAME";
               break;
      case -47: error_message = "CL_INVALID_KERNEL_DEFINITION";
               break;
      case -48: error_message = "CL_INVALID_KERNEL";
               break;
      case -49: error_message = "CL_INVALID_ARG_INDEX";
               break;
      case -50: error_message = "CL_INVALID_ARG_VALUE";
               break;
      case -51: error_message = "CL_INVALID_ARG_SIZE";
               break;
      case -52: error_message = "CL_INVALID_KERNEL_ARGS";
               break;
      case -53: error_message = "CL_INVALID_WORK_DIMENSION";
               break;
      case -54: error_message = "CL_INVALID_WORK_GROUP_SIZE";
               break;
      case -55: error_message = "CL_INVALID_WORK_ITEM_SIZE";
               break;
      case -56: error_message = "CL_INVALID_GLOBAL_OFFSET";
               break;
      case -57: error_message = "CL_INVALID_EVENT_WAIT_LIST";
               break;
      case -58: error_message = "CL_INVALID_EVENT";
               break;
      case -59: error_message = "CL_INVALID_OPERATION";
               break;
      case -60: error_message = "CL_INVALID_GL_OBJECT";
               break;
      case -61: error_message = "CL_INVALID_BUFFER_SIZE";
               break;
      case -62: error_message = "CL_INVALID_MIP_LEVEL";
               break;
      case -63: error_message = "CL_INVALID_GLOBAL_WORK_SIZE";
               break;
      case -64: error_message = "CL_INVALID_PROPERTY";
               break;
      case -65: error_message = "CL_INVALID_IMAGE_DESCRIPTOR";
               break;
      case -66: error_message = "CL_INVALID_COMPILER_OPTIONS";
               break;
      case -67: error_message = "CL_INVALID_LINKER_OPTIONS";
               break;
      case -68: error_message = "CL_INVALID_DEVICE_PARTITION_COUNT";
               break;

      // extension errors
      case -1000: error_message = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
               break;
      case -1001: error_message = "CL_PLATFORM_NOT_FOUND_KHR";
               break;
      case -1002: error_message = "CL_INVALID_D3D10_DEVICE_KHR";
               break;
      case -1003: error_message = "CL_INVALID_D3D10_RESOURCE_KHR";
               break;
      case -1004: error_message = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
               break;
      case -1005: error_message = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
               break;
      default: error_message = "Unknown OpenCL error";
   }

   return error_message;
}

// Handles an error from OpenCL
void error_handler(cl_int err, std::string message = "") {

   if(err == CL_SUCCESS)
      return;

   std::string error_message = get_error_message(err);

   std::cout << error_message << std::endl;
   if(!message.empty()){
      std::cout << message << std::endl;
   }
   exit(1);
}

//Handles a vector of errors from OpenCL and clears the vector when done
void error_handler(std::vector<cl_int> & errors, std::string message = ""){
   bool noneBad = true;
   for(size_t i=0; i<errors.size(); ++i){
      if(errors[i] == CL_SUCCESS) continue;
      else noneBad = false;

      std::string error_message = get_error_message(errors[i]);

      std::cout << error_message << std::endl;
      if(!message.empty()){
         std::cout << message << std::endl;
      }
   }

   if(noneBad){
      errors.clear();
      return;
   }

   exit(1);
}

#endif /* error_handler.h */