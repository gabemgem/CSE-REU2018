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


//Not sure this will work because due to unpredictable runtime ordering of PE's
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

__kernel void newLineAlt(__global char * input, __global uint * output, uint size){
   uint gid = get_global_id(0);
   if(gid < size){
      output[gid] = (input[gid] == NEWLINE);
   }
}

__kernel void getLinePos(__global uint * data, __global uint * output, uint size){
   uint gid = get_global_id(0);
   if(gid < size){
      if(gid == 0){
         if(data[0]){
            output[1] = 0;
            output[2] = 1;
         }
      }
      else if(data[gid] != data[gid-1]){
         output[(2*data[gid-1]) + 1] = gid;
         output[2*data[gid]] = gid + 1;
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
   __global uint* result_sizes,  //sizes of the final result for each line
   __local char* lstring,        //array to hold the local string
   __local char* escape,         //array to hold locations of escape characters
   __local char* function,       //array to calculate the function
   uint lines,                   //number of lines in input_string
   __local uint* len,            //length of current line
   __local uint* curr_pos,       //holds copy of the current line pointer for work group
   __local char* prev_escape,    //holds the escape value of the last element in for previous buffer
   __local char* prev_function,  //holds the function value of the last element in for previous buffer
   __local uint* prev_sep,       //holds the separators value of the last element in for previous buffer
   __local uint* elems_scanned,  //tracks the number if elements scanned
   __local char* first_char      //denotes first character of a line is delimited
	) {
   
   uint gid = get_global_id(0), lid = get_local_id(0);
   uint gw_size = get_global_size(0), wg_size = get_local_size(0);

	//compute until all lines are exhausted
	while(atomic_add(pos_ptr, 0) < lines){
      
      //setting up for new line
      if(lid == 0){
			*curr_pos = atomic_inc(pos_ptr) * 2;
			*len = input_pos[(*curr_pos) + 1] - input_pos[*curr_pos];
			*first_char = (input_string[input_pos[*curr_pos]] == OPEN);

			*prev_escape = 0;
			*prev_function = IDENTITY;
			*prev_sep = 0;
         *elems_scanned = 0;
		}
      barrier(CLK_LOCAL_MEM_FENCE);

      while((*elems_scanned) < (*len)){

         //copy elements of input string from global memory to local and set function
         uint index = input_pos[*curr_pos] + (*elems_scanned) + lid;
         lstring[lid] = (((*elems_scanned) + lid) < (*len)) ? input_string[index] : ' ';
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
                           !((*first_char) ? ((function[lid] & 2) >> 1) : (function[lid] & 1));
         barrier(CLK_LOCAL_MEM_FENCE);

         //parallel add over separators elements
         parScanAdd(separators, wg_size);
         separators[lid] += *prev_sep;

         //copy final result to global memory and updating result size
         if(lid == 0){
            if(*prev_sep != separators[lid]){
               finalResults[input_pos[*curr_pos] + *prev_sep] = index;
               atomic_inc(&(result_sizes[(*curr_pos)/2]));
            }
         }
         else if(separators[lid] != separators[lid-1]){
               finalResults[input_pos[*curr_pos] + separators[lid-1]] = index;
               atomic_inc(&(result_sizes[(*curr_pos)/2]));
         }
         
         //save results of last element
         //increase elems_scanned by the work group size
         if(lid == wg_size - 1) {
			   *prev_escape = escape[lid];
			   *prev_function = function[lid];
			   *prev_sep = separators[lid];
			   (*elems_scanned) += wg_size;
		   }

         barrier(CLK_LOCAL_MEM_FENCE);
      }

   }

}


__kernel void flipCoords(
   __global char* input_string,     //Just the polyline from original string
   __global uint* start_positions,  //Array of comma locations between pairs
   __global uint* num_pairs,        //Number of pairs in polyline
   __global uint* pos_ptr,          //WG's atomically increment to choose pair
   __local uint* curr_pos,          //Which pair WG is currently looking at
   __local uint* loc_length,        //Length of local pair
   __local char* str,               //String to copy local data into
   __local uint* mid,               //Holds location of the comma in a pair
   __local uint* y_len,             //Holds length of y coord for the pair
   __global char* output_string     //Polyline output
      ) {

   uint gid = get_global_id(0), lid = get_local_id(0);
   uint wg_size = get_local_size(0);
   while(*pos_ptr<*num_pairs) {
      if(lid==0) {
         *curr_pos = atomic_inc(pos_ptr);
         *loc_length = start_positions[*curr_pos+1]-start_positions[*curr_pos];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for(uint i = 0; i<*loc_length; i+=wg_size) {
         if(i+lid<*loc_length && input_string[i+lid+start_positions[*curr_pos]] == SEP) {
            *mid = lid+i;
            *y_len = *loc_length - *mid;
            output_string[start_positions[*curr_pos]+*y_len] = ',';
            break;
         }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for(uint i = 0; i<*loc_length; i+=wg_size) {
         if(lid+i!=*mid && lid+i<*loc_length) {
            uint target = (lid+i>*mid) ? lid + i - *mid : *y_len + lid + i + 1;
            output_string[target] = input_string[start_positions[*curr_pos]+i+lid];
         }
      }
   }
   
}