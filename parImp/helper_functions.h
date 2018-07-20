#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_int pad_num(cl_int old) {
   cl_int new = 1;
   while(old>new) {
      new<<=1;
   }
   return new;
}

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

cl_uint read_from_file(FILE* fp, char* line, cl_int* guess, char* eof) {
   if(!fp) {
      printf("Couldn't open input file");
      exit(1);
   }

   cl_uint counter=0, size=*guess;
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
         if(line==NULL) {
            perror("Couldn't realloc line");
            exit(1);
         }
      }

   }
   if(counter<size) {
      for(int i = counter; i < size; ++i) {
         line[i] = ' ';
      }
   }
   *guess = counter;
   *guess = pad_num(*guess);
   
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