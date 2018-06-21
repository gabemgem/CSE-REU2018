#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


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


#endif /* helper_functions.h */