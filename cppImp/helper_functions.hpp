#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <CL/cl.hpp>

#define CHUNK_SIZE 2048


/* Reads in a chunk of data from file. Ensures that  the chunk
   starts/ends on with a complete line.
*/
void read_chunk(std::ifstream & file, std::string & chunk, std::string & residual){
   if(!file.is_open()) {
      exit(1);
   }
   unsigned int size = chunk.size();
   std::string line;
   while(std::getline(file, line)){
      if(size + line.size() > CHUNK_SIZE){
         //removing ending newline
         //chunk = chunk.substr(0, size-1);
         residual = line;
         file.close();
         return;
      }

      chunk += line + "\n";
      size = chunk.size();
   }
   file.close();
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

cl_uint lg(cl_uint val){
   cl_uint out = 0, cpy = val;
   while(cpy > 1){
      cpy >>= 1;
      ++out;
   }
   if((1 << out) < val) ++out;

   return out;
}

#endif /* helper_functions.h */