#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

#include <stdio.h>
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

/* Copies a chuck of data from a file close to CHUNK_SIZE that ends in a '\n' */
cl_uint read_chunck(FILE * fp, char ** chunk, char ** residual, unsigned int len){
   cl_char c;
   unsigned int count = len;
   unsigned long start = ftell(fp);
   
   //copying chunk from file
   while((c = fgetc(fp)) != EOF || count < CHUNK_SIZE){
      *chunk = (char *)realloc(*chunk, (++count)*sizeof(cl_char));
      (*chunk)[count-1] = c;
   }

   if(c == EOF){
      *residual = NULL;
      return count;
   }

   if((*chunk)[count-1] != '\n'){
      //seeking the past '\n'
      unsigned int end = count-1;
      while(c != '\n'){
         --end;
         c = (*chunk)[end];
      }
      
      unsigned int chunk_size = count - end;
      unsigned int res_size = count - chunk_size;
      
      //copying start of next line into residual
      *residual = (char *)malloc(sizeof(cl_char)*(res_size));
      mempcpy(*residual, (*chunk) + chunk_size, res_size);

      //Getting rid of incomplete line from chunk and throwing away '\n'
      *chunk = (char*)realloc(*chunk, sizeof(cl_char)*(chunk_size-1));

      return chunk_size;
   }

   //throwing away ending '\n'
   *chunk = (char*)realloc(*chunk, sizeof(cl_char)*(count-1));
   return count-1;

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