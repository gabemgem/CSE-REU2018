#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'
#define NEWLINE '\n'

//The identity for boolean function composition
#define IDENTITY 2

/* Function to compute g(f) */
inline char compose(char f, char g) {
	char h = 0;

	h |= (g & (1 << (f & 1))) >> (f & 1);

	h |= (g & 1 << ((f&2) >> 1)) << (1 - ((f & 2) >> 1));

	return h;
}


/* Finds newline characters and marks them */
__kernel void newLine(__global char * input, 
                      __global uint * output,//Mark location of newlines with 1's 
                      uint size,             //Size of input
		                __global int* out_pos, //Array of newline locations in no particular order
                      __global uint* pos_ptr,//uint to keep track of index to put location into
                      uint nlines            //Number of lines to read
){
   uint gid = get_global_id(0);
   uint gsize = get_global_size(0);
   for(uint i = 0; i < size; i+=gsize) {
	   if(gid+i < size){
	      output[gid+i] = (input[gid+i] == NEWLINE);
	      
	      if(input[gid+i] == NEWLINE) {
	      	out_pos[atomic_inc(pos_ptr)]=gid+i;
	      	
	      	if((*pos_ptr)>nlines) {
	      		return;
	      	}
	      }
	   }
	}
}

/* Marks the position of newline charaters */
__kernel void newLineAlt(__global char * input, 
                         __global uint * output,//Mark location of newlines with 1's
                         uint size              //Size of input
){
   uint gid = get_global_id(0);
   uint gsize = get_global_size(0);
   for(uint i = 0; i < size; i+=gsize) {
      if(gid+i < size){
         output[gid+i] = (input[gid+i] == NEWLINE);
      }
   }
}

/* 
   Records the start/end of each line after performing a
   parallel scan add on the output of newLineAlt;
   assumed that first elem is 0 and the last is "chunk size" - 1 
*/
__kernel void getLinePos(__global uint * data, __global uint * output, uint size){
   uint gid = get_global_id(0);
   uint gsize = get_global_size(0);
   for(uint i = 0; i < size; i+=gsize) {
      if(gid+i < size){
         if(gid+i == 0){
            if(data[0]){
               output[1] = 0;
               output[2] = 1;
            }
         }
         else if(data[gid+i] != data[gid+i-1]){
            output[(2*data[gid+i-1]) + 1] = gid+i;
            output[2*data[gid+i]] = gid+i + 1;
         }
      }
   }
}

/* 
   addScanStep and addPostScanStep used only for finding the postions
   of lines in data chunck 
*/
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

/* Performs a parallel scan composition of the delimiter
   finding function */
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

/* Performs a parallel scan add */
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
   (ex: setting position and saving previous values) 

   Kernel to find the separators in an input string
   Each workgroup takes a line to parse in a queue-like
   manner.
*/
__kernel void findSep(
   __global char *input_string,  //array with the input
   __global uint *input_pos,     //array of start/end position pairs for each line
   __global uint *pos_ptr,       //points to a position pair in input_pos
   __local uint *separators,     //array for valid separators
   __global uint *finalResults,  //array to hold final scan results
   __global uint *result_sizes,  //sizes of the final result for each line
   __local char *lstring,        //array to hold the local string
   __local char *escape,         //array to hold locations of escape characters
   __local char *function,       //array to calculate the function
   uint lines                    //number of lines in input_string
   ) {
   
   uint gid = get_global_id(0), lid = get_local_id(0);
   uint gw_size = get_global_size(0), wg_size = get_local_size(0);

   __local uint len;             //length of current line
   __local uint curr_pos;        //holds copy of the current line pointer for work group
   __local char prev_escape;     //holds the escape value of the last element in for previous buffer
   __local char prev_function;   //holds the function value of the last element in for previous buffer
   __local uint prev_sep;        //holds the separators value of the last element in for previous buffer
   __local uint elems_scanned;   //tracks the number if elements scanned
   __local char first_char;      //denotes first character of a line is delimited

	//compute until all lines are exhausted
	while(atomic_add(pos_ptr, 0) < lines){
      
      //setting up for new line
      if(lid == 0){
			curr_pos = atomic_inc(pos_ptr) * 2;
			len = input_pos[(curr_pos) + 1] - input_pos[curr_pos];
			first_char = (input_string[input_pos[curr_pos]] == OPEN);

			prev_escape = 0;
			prev_function = IDENTITY;
			prev_sep = 0;
         elems_scanned = 0;
		}
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //parsing line
      while((elems_scanned) < (len)){

         //copy elements of input string from global memory to local and set function
         uint index = input_pos[curr_pos] + elems_scanned + lid;
         lstring[lid] = ((elems_scanned + lid) < len) ? input_string[index] : ' ';
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
         parScanCompose(function, wg_size);
         function[lid] = compose(prev_function, function[lid]);
         
         //initialize separators for characters in buffer
         separators[lid] = (lstring[lid] == SEP) &&
                           !((first_char) ? ((function[lid] & 2) >> 1) : (function[lid] & 1));
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel add over separators elements
         parScanAdd(separators, wg_size);
         separators[lid] += prev_sep;

         //copy final result to global memory and updating result size
         if(lid == 0){
            if(prev_sep != separators[lid]){
               finalResults[input_pos[curr_pos] + prev_sep] = index;
               atomic_inc(&(result_sizes[curr_pos/2]));
            }
         }
         else if(separators[lid] != separators[lid-1]){
               finalResults[input_pos[curr_pos] + separators[lid-1]] = index;
               atomic_inc(&(result_sizes[curr_pos/2]));
         }

         //save results of last element
         //increase elems_scanned by the work group size
         if(lid == wg_size - 1) {
			   prev_escape = escape[lid];
			   prev_function = function[lid];
			   prev_sep = separators[lid];
			   elems_scanned += wg_size;
		   }

         barrier(CLK_LOCAL_MEM_FENCE);
      }

   }

}


/* 
   Kernel to flip coordinates in the polyline
   Each workgroup handles a single pair of coordinates
   Only flips coordinates in one line at a time
   This is helps minimize the irregularity
*/
__kernel void flipCoords(
   __global char* input_string,     //The original string
   __global uint* start_positions,  //Array of comma locations between pairs
   __global uint* pos_ptr,          //WG's atomically increment to choose pair
   __global char* output_string,    //Polyline output
   uint num_pairs,                  //Number of pairs in polyline
   uint finalSize,                  //size of output_string
   uint currStart                   //Start of polyline
   ) {

   
   uint gid = get_global_id(0), lid = get_local_id(0);
   uint glob_size = get_global_size(0), wg_size = get_local_size(0);
   
   //Copy current part of input_string to output_string
   for(uint i = 0; i < finalSize; i += glob_size) {
      if(gid+i<finalSize) {
         char result = input_string[start_positions[currStart] + gid+i + 1];
         output_string[gid+i] = result;
      }
   }

   //Local variables
   __local uint loc_start; //position of the start of current pair
   __local uint loc_end;   //position of the end of current pair
   __local uint curr_pos;  //current pair to look at
   __local uint loc_length;//length of current pair
   __local uint mid;       //middle comma location
   __local uint y_len;     //length of y coordinate
   __local uint lineStart; //start position of current line

   //loop through all pairs
   while(atomic_add(pos_ptr, 0)<num_pairs) {
      //set up local variables
      if(lid==0) {
         curr_pos = atomic_inc(pos_ptr);
         if(curr_pos >= num_pairs) return;
         
         lineStart = start_positions[currStart];

         loc_start = start_positions[currStart + curr_pos];
         loc_start += ((!curr_pos) ? 4 : 3);

         loc_end = (curr_pos == num_pairs - 1) ? lineStart + finalSize - 2
                     : start_positions[currStart + curr_pos + 1] - 1;
         
         loc_length = loc_end - loc_start;

         mid=0;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //find middle comma in current pair
      for(uint i = 0; i<loc_length; i+=wg_size) {
         uint index = loc_start + lid + i;
         if(index < loc_end){
            if(input_string[index] == SEP){
               mid = index;
               y_len = loc_length - (mid - loc_start) - 2;
               output_string[loc_start + y_len - lineStart - 1] = ',';
               output_string[loc_start + y_len - lineStart] = ' ';
            }
         }
      }
      
      
      barrier(CLK_LOCAL_MEM_FENCE); 
      
      //flip coordinates in current pair
      for(uint i = 0; i<loc_length; i+=wg_size) {
         uint index = loc_start + lid + i;
         if(index != mid && index < loc_end) {
            uint target;
            if(index > mid + 1){//y coordinate
               target = loc_start + (index - mid - 1) - lineStart - 2;
            }
            if(index < mid){//x coordinate
               target = index - lineStart - 1 + y_len + 2;
               
            }
            output_string[target] = input_string[index];
         }
      }
      
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}