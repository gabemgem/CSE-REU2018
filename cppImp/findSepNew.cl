#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'
#define NEWLINE '\n'

inline char compose(char f, char g) {
	char h = 0;

	h |= (g & (1 << (f & 1))) >> (f & 1);

	h |= (g & 1 << ((f&2) >> 1)) << (1 - ((f & 2) >> 1));

	return h;
}

/* Finds newline characters and marks them*/
__kernel void newLine(__global char * input, __global uint * output){
   uint gid = get_local_id(0);
   output[gid] = input[gid] == NEWLINE;
}

/* NOTE: It may be slightly more efficient to use atomic function to 
   select the first PE to respond to do single-person tasks
   (ex: setting position and saving previous values) */
__kernel void findSep(
   __global char* input_string,  //char array with the input
   __global uint* pos,           //uint array of start/end position pairs for each line
   __global uint lines,          //number of lines in input_string
   __global uint* pos_ptr,       //points to a position pair in pos
   __local uint length,          //length of current line
   __local uint curr_pos,        //holds copy of the current line pointer for work group
   __local char* lstring,        //char array to hold the local string
   __local char* escape,         //array to hold locations of escape characters
   __local prev_escape,          //holds the escape value of the last element in for previous buffer
   __local prev_function,        //holds the function value of the last element in for previous buffer
   __local prev_val,             //holds the separator value of the last element in for previous buffer
   __local char* function,       //array to calculate the function
   __local char elems_scanned,   //tracks the number if elements scanned
   __local char first_char,      //denotes first character of a line is delimited
   __local uint* separators,     //array for valid separators
   __global uint* finalResults   //array to hold final scan results
	) {
   
   uint gid = get_global_id(0), lid = get_local_id(0);
   uint gw_size = get_global_size(0), wg_size = get_local_size(0);

   //compute until all lines are exhausted
   while(atomic_add(pos_ptr, 0) < lines){
      
      //setting up for new line
      if(lid == 0){
         curr_pos = atomic_inc(pos_ptr) * 2;
         length = pos[curr_pos + 1] - pos[curr_pos];
         first_char = (input_string[pos[curr_pos]] == OPEN);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      while(elems_scanned < length){

         //copy elements of input string from global memory to local and set function
         uint index = curr_pos + elems_scanned + lid;
         lstring[lid] = (index < length) ? input_string[index] : ' ';
         barrier(CLK_LOCAL_MEM_FENCE);

         //initialize function for characters in buffer
         char open = (lstring[lid] == OPEN);
         char close = (lstring[lid] == CLOSE);
         escape[lid] = (lstring[lid] == ESC);

         function[lid] = open;
         function[lid] |= (lid != 0) ? ((!close) || escape[lid-1]) << 1 : 
                                       ((!close) || prev_escape) << 1;
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel compose over function elements
         parScanCompose(function, wgsize);
         function[lid] = compose(last_function, function[lid]);
         
         //initialize separators for characters in buffer
         separators[lid] = (first_char) ? (function & 2) >> 1) :
                                          (function & 1);
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel add over separator elements
         parScanAdd(separators, wgsize);
         separators[lid] += prev_val;

         //save result of last element
         if(lid == wg_size - 1) {
			   prev_escape = escape[lid];
			   prev_function = function[lid];
			   prev_val = separators[lid];
		   }

         //increase elems_scanned by the work group size
         elems_scanned += ((lid == 0) ? wg_size : 0);

         barrier(CLK_LOCAL_MEM_FENCE);
      }

   }

}