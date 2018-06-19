#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#define OPEN '['
#define CLOSE ']'
#define SEP ','
#define ESC '\\'

char asso_func(char f, char g){
  char _f[] = {f & 1, (f & 2) >> 1};
  char _g[] = {g & 1, (g & 2) >> 1};

  char h = 0;
  h |= _g[_f[0]];
  h |= _g[_f[1]] << 1;
  
  return h;
}

int lg(unsigned int val){
  int out = 0;
  while(val > 1){
    val >>= 1;
    ++out;
  }
  return out;
}

void scanOp(char* data, int len){
  //scan step
  int depth = lg(len);
  for(int d=0; d<depth; ++d){
    int mask = (0x1 << d) - 1;
    for(int i=0; i < len; ++i){
      int ind1 = (i*2)+1;
      if(((i & mask) == mask) && (ind1 < len)){
	int offset = 1 << d;
	int ind0 = ind1 - offset;
	data[ind1] = asso_func(data[ind0], data[ind1]);
      }
    }
  }
  
  //post inclusive step
  for(int stride = len/4; stride > 0; stride /= 2){
    for(int i=0; i < len; ++i){
      int ind = (2*stride*(i + 1)) - 1;
      if(ind + stride < len){
	data[ind + stride] = asso_func(data[ind], data[ind + stride]);
      }
    }
  }
}

char* init_function(char* str, char* escape, int len){
  char* func = malloc(sizeof(char)*len);
  for(int i=0; i<len; ++i){
    char open = str[i] == OPEN,
      close = str[i] == CLOSE;

    escape[i] = str[i] == ESC;
    
    func[i] = 0;
    func[i] |= open;
    if(i != 0){
      func[i] |= ((!close) | escape[i-1]) << 1;
    }
    else{
      func[i] |= (!close) << 1;
    }
  }

  return func;
}


int main(int argc, char ** argv){
  if(argc != 2){
    printf("Need 1 arguments passed\n");
    exit(-1);
  }
  
  char str[128];
  strcpy(str, argv[1]);
  int len = strlen(str);

  char* escape = malloc(sizeof(char)*len);
  char* seperator = malloc(sizeof(char)*len);
  
  char* function = init_function(str, escape, len);
  
  printf("%s\n", str);
  
  scanOp(function, len);
  
  char firstChar = str[0] == OPEN;

  for(int i=0; i < len; ++i){
    seperator[i] = (str[i] == SEP) && !(function[i] & (1 << firstChar));
  }
  
  for(int i=0; i < len; ++i){
    printf("%d", seperator[i]);
  }
  printf("\n");
  
  return 0;
}
