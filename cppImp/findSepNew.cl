#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'
#define NEWLINE '\n'
#define IDENTITY 2

inline char compose(char f, char g) {
	char h = 0;

	h |= (g & (1 << (f & 1))) >> (f & 1);

	h |= (g & 1 << ((f&2) >> 1)) << (1 - ((f & 2) >> 1));

	return h;
}

/* Finds newline characters and marks them*/
__kernel void newLine(__global char * input, __global uint * output, uint size,
		__global int* out_pos, __global uint* pos_ptr, uint nlines){
   uint gid = get_global_id(0);
   uint gsize = get_global_size(0);
   for(uint i = 0; i < size; i+=gsize) {
	   if(gid+i < size){
	      output[gid+i] = (input[gid+i] == NEWLINE);
	      //out_pos[gid+i] = (input[gid+i] == NEWLINE) ? gid+i : 0;
	      
	      if(input[gid+i] == NEWLINE) {
	      	out_pos[atomic_inc(pos_ptr)]=gid+i;
	      	
	      	if((*pos_ptr)>nlines) {
	      		return;
	      	}
	      }
	      
	   }
	}
}

/* addScanStep and addPostScanStep used only for finding the postions
   of lines in data chunck */
__kernel void addScanStep(__global uint* data, uint size, uint d){
   uint gid = get_global_id(0);

   int mask = (0x1 << d) - 1;
   char selected = ((gid & mask) == mask) && (gid < size/2);
   
   uint ind1 = (selected) ? (gid*2)+1 : 0;
   uint offset = 0x1 << d;
   uint ind0 = (selected) ? ind1 - offset : 0;
   
   uint h = data[ind0] + data[ind1];
   data[ind1] = (selected) ? h : data[ind1];
}

__kernel void addPostScanStep(__global uint* data, uint size, uint stride){
   uint gid = get_global_id(0);

   uint ind = (2*stride*(gid + 1)) - 1;
   uint ind2 = ind + stride;
   char selected = ind2 < size;
   
   ind = (selected) ? ind : 0;
   ind2 = (selected) ? ind2 : 0;

   uint h = data[ind] + data[ind2];
   data[ind2] = (selected) ? h : data[ind2];
}

inline void parScanCompose(__local char* func, uint size){
   uint lid = get_local_id(0);
   uint ind1 = (lid*2)+1;
   uint depth = log2((float)size);
   
   //scan step
   for(uint d=0; d<depth; ++d){
      uint mask = (0x1 << d) - 1;
      if(((lid & mask) == mask) && (lid < size/2)){
         uint offset = 0x1 << d;
         uint ind0 = ind1 - offset;
         func[ind1] = compose(func[ind0], func[ind1]);
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   //post scan step
   for(uint stride = size/4; stride > 0; stride /= 2){
      uint ind = (2*stride*(lid + 1)) - 1;
      uint ind2 = ind + stride;
      if(ind2 < size){
         func[ind2] = compose(func[ind], func[ind2]);
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

}

inline void parScanAdd(__local uint* data, uint size){
   uint lid = get_local_id(0);
   uint ind1 = (lid*2)+1;
   uint depth = log2((float)size);
   
   //scan step
   for(uint d=0; d<depth; ++d){
      uint mask = (0x1 << d) - 1;
      if(((lid & mask) == mask) && (lid < size/2)){
         uint offset = 0x1 << d;
         uint ind0 = ind1 - offset;
         data[ind1] += data[ind0];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   //post scan step
   for(uint stride = size/4; stride > 0; stride /= 2){
      uint ind = (2*stride*(lid + 1)) - 1;
      uint ind2 = ind + stride;
      if(ind2 < size){
         data[ind2] += data[ind];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

}


/* NOTE: It may be slightly more efficient to use atomic function to 
   select the first PE to respond to do single-person tasks
   (ex: setting position and saving previous values) */
__kernel void findSep(
   __global char* input_string,  //array with the input
   __global uint* input_pos,     //array of start/end position pairs for each line
   __global uint* pos_ptr,       //points to a position pair in input_pos
   __local uint* separators,     //array for valid separators
   __global uint* finalResults,  //array to hold final scan results
   __global uint* result_sizes,   //size of the final result for each line
   __local char* lstring,        //array to hold the local string
   __local char* escape,         //array to hold locations of escape characters
   __local char* function,       //array to calculate the function
   uint lines,          //number of lines in input_string
   uint len,          //length of current line
   __local uint* curr_pos,        //holds copy of the current line pointer for work group
   __local char* prev_escape,          //holds the escape value of the last element in for previous buffer
   __local char* prev_function,        //holds the function value of the last element in for previous buffer
   __local uint* prev_sep,             //holds the separators value of the last element in for previous buffer
   __local char* elems_scanned,   //tracks the number if elements scanned
   __local char* first_char      //denotes first character of a line is delimited
	) {
   
   uint gid = get_global_id(0), lid = get_local_id(0);
   uint gw_size = get_global_size(0), wg_size = get_local_size(0);

	//compute until all lines are exhausted
	while(atomic_add(pos_ptr, 0) < lines){
      
      //setting up for new line
      if(lid == 0){
			*curr_pos = atomic_inc(pos_ptr) * 2;
			if(*curr_pos>=lines*2) {
		   		return;
			}
			len = input_pos[*curr_pos + 1] - input_pos[*curr_pos];
			*first_char = (input_string[input_pos[*curr_pos]] == OPEN);

			*prev_escape = 0;
			*prev_function = IDENTITY;
			*prev_sep = 0;
		}
      barrier(CLK_LOCAL_MEM_FENCE);

      while(*elems_scanned < len){

         //copy elements of input string from global memory to local and set function
         uint index = *curr_pos + *elems_scanned + lid;
         lstring[lid] = (index < len) ? input_string[index] : ' ';
         barrier(CLK_LOCAL_MEM_FENCE);

         //initialize function for characters in buffer
         char open = (lstring[lid] == OPEN);
         char close = (lstring[lid] == CLOSE);
         escape[lid] = (lstring[lid] == ESC);

         function[lid] = open;
         function[lid] |= (lid != 0) ? ((!close) || escape[lid-1]) << 1 : 
                                       ((!close) || *prev_escape) << 1;
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel compose over function elements
         parScanCompose(function, wg_size);
         function[lid] = compose(*prev_function, function[lid]);
         
         //initialize separators for characters in buffer
         separators[lid] = (lstring[lid] == SEP) &&
                           (*first_char) ? ((function[lid] & 2) >> 1) : (function[lid] & 1);
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel add over separators elements
         parScanAdd(separators, wg_size);
         separators[lid] += *prev_sep;

         //copy final result to global memory and updating result size
         if(lid == 0){
            if(*prev_sep != separators[lid]){
               finalResults[*curr_pos + separators[lid]] = index;
               atomic_inc(&result_sizes[*curr_pos/2]);
            }
         }
         else if(separators[lid] != separators[lid-1]){
               finalResults[*curr_pos + separators[lid]] = index;
               atomic_inc(&result_sizes[*curr_pos/2]);
         }

         //save results of last element
         //increase elems_scanned by the work group size
         if(lid == wg_size - 1) {
			   *prev_escape = escape[lid];
			   *prev_function = function[lid];
			   *prev_sep = separators[lid];
			   *elems_scanned += wg_size;
		   }

         barrier(CLK_LOCAL_MEM_FENCE);
      }

   }

}