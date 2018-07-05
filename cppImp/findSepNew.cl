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

inline void initFunction()

__kernel void findSep(
	__global char* input_string,     //char array with the input
	__global uint* pos,              //uint array of start/end position pairs for each line
   __global uint lines,             //number of lines in input_string
   __global uint* pos_ptr,          //points to a position pair in pos
   __local uint* isWorking          //denotes when work is being done on a current line
   __local uint* curr_pos,          //holds copy of the current line pointer for work group
	__local char* lstring,           //char array to hold the local string
	__local char* escape,            //array to hold locations of escape characters
	__local char* function,          //array to calculate the function
   __local char* elems_scanned,     //tracks the number if elements scanned
   __local char* first_character    //first character of a line
	__local uint* separators         //array for valid separators
	__global uint* finalResults      //array to hold final scan results
	) {
   
   //compute until all lines are exhausted
   while(atomic_add(pos_ptr, 0) < lines){
      
      //setting position pointer
      if(!atomic_inc(isWorking)){
         *curr_pos = atomic_inc(pos_ptr);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      


      barrier(CLK_LOCAL_MEM_FENCE);
   }

}

__kernel void findSep(
	__global char* input_string//char array with the input
	__global uint* line_lengths//uint array of line lengths
	__local char* lstring//char array to hold the local string
	__local char* escape//array to hold locations of escape characters
	__local char prev_escape//char to hold previous escape value
	__local char* function//array to calculate the function
	__local char prev_function//char to hold previous function value
	__local char first_character//char to hold previous first character value for function
	__local uint* separators//array for valid separators
	__global uint* finalResults//array to hold final scan results
	) {

	uint gid = get_global_id(0);//global id
	uint lid = get_local_id(0);//local id
	uint wgid = get_group_id(0);//work group id
	uint wgsize = get_local_size(0);//work group size
	uint llength = line_lengths[wgid];//work group line length
	
   if(lid==0) {
		prev_escape = 0;
		first_character = input_string[gid]==OPEN;
		prev_function = 2;
	}

	for(uint i = 0; i < llength; i+=wgsize) {
		lstring[lid] = (lid+i<llength) ? input_string[gid+i] : '0';

		char open = (lstring[lid] == OPEN);
		char close = (lstring[lid] == CLOSE);
		escape[lid] = (lstring[lid] == ESC);

		function[lid] = 0;
		function[lid] |= open;
		barrier(CLK_LOCAL_MEM_FENCE);
		function[lid] |= (lid!=0) ? ((!close) || escape[lid-1]) << 1 :
									((!close) || prev_escape) << 1;

		parScanCompose(function, wgsize);
		compose(prev_function, function[lid]);

		separators[lid] = (lstring[lid] == SEP) &&
			!(function[lid] & (1<<first_character));

		parScanAdd(separators, wgsize);
		separators[lid]+=prev_separators;

		if(lid == wgsize-1) {
			prev_escape = escape[lid];
			prev_function = function[lid];
			prev_separators = separators[lid];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid+i<llength) {
			finalResults[gid+1] = separators[lid];
		}



	}
}