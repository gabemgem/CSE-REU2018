#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

//#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <CL/cl.hpp>

#define CHUNK_SIZE 1024

/* Returns the next power of 2 from old */
cl_int pad_num(cl_int old) {
   cl_int new_val = 1;
   while(old > new_val) {
      new_val <<= 1;
   }
   return new_val;
}

/* Reads in a chunk of data from file. Ensures that  the chunk
   starts/ends on with a complete line.
*/
void read_chunk(FILE * fp, char ** chunk, char ** residual,
               cl_uint *len, cl_uint *residual_len){
   char c;
   cl_uint count = *len;
   
   //copying chunk from file
   while(count < CHUNK_SIZE){
      
      //hit the end of file
      if((c = fgetc(fp)) == EOF){
         //getting rid of any possible ending newlines
         while((*chunk)[count-1] == '\n'){
            char* nchunk = (char *)realloc(*chunk, (--count)*sizeof(char));
            if(!nchunk) {
               exit(1);
            }
            *chunk = nchunk;
         }
         *residual = NULL;
         *residual_len = 0;
         *len = count;
         return;
      }

      char* nchunk = (char *)realloc(*chunk, (++count)*sizeof(char));
      if(!nchunk) {
         exit(1);
      }
      *chunk = nchunk;
      (*chunk)[count-1] = c;
   }

   //seeking the past '\n'
   if(c != '\n'){
      cl_uint chunk_size = count-1;
      while(c != '\n'){
         --chunk_size;
         c = (*chunk)[chunk_size];
      }
      
      cl_uint res_size = count - chunk_size - 1;
      
      //copying start of next line into residual
      *residual = (char *)malloc(sizeof(char)*(res_size));
      memcpy(*residual, (*chunk) + chunk_size + 1, sizeof(char)*(res_size));

      //Getting rid of incomplete line from chunk and throwing away '\n'
      char* nchunk = (char*)realloc(*chunk, sizeof(char)*(chunk_size));
      if(!nchunk) {
         exit(1);
      }
      *chunk = nchunk;

      *len = chunk_size;
      *residual_len = res_size;
      return;
   }

   //throwing away ending '\n'
   char* nchunk = (char*)realloc(*chunk, sizeof(char)*(count-1));
   if(!nchunk) {
      exit(1);
   }
   *chunk = nchunk;
   *residual = NULL;
   *residual_len = 0;
   *len = count-1;
   return;
}

/* Reads in a chunk of data from file. Ensures that  the chunk
   starts/ends on with a complete line.
*/
void read_chunk_pp(std::ifstream & file, std::string & chunk, std::string & residual){
   unsigned int size = chunk.size();
   std::string line;
   while(std::getline(file, line)){
      if(size + line.size() > CHUNK_SIZE){
         //removing ending newline
         chunk = chunk.substr(0, size-1);
         residual = line;
         return;
      }

      chunk += line + "\n";
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

/* Padding string w/ spaces to a length of next power of 2. */
void pad_string(char** str, cl_int* len){ 
   //returns the next power of 2 from len
   cl_int new_len = pad_num(*len);

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

#endif /* helper_functions.h */