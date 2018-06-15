//function to compose 2-variable boolean functions
//functions represented as two bits of a char
//h = g(f)
//the identity is defined by the bits 0b10 (i.e. 2)
inline char compose(char f, char g) {
   
   char h = 0;
   //puts g(f(0)) into the first bit of h
   h |= (g & (1 << (f & 1))) >> (f & 1);
   //puts g(f(1)) into the second bit of h
   h |= (g & 1 << ((f&2) >> 1)) << (1 - ((f & 2) >> 1));

   return h;
}

//sweepup stage of parallel scan
inline void sweepup1(__local uint* x, int m) {
   int lid = get_local_id(0);
   int ind1 = (lid*2)+1;//g function location
   int depth = log2(m);
   for(int d=0; d<depth; ++d) {
      barrier(CLK_LOCAL_MEM_FENCE);//sync work items
      int mask = (0x1 << d) - 1;
      if((lid & mask) == mask) {//mask unused work items
         int offset = 0x1 << d;
         int ind0 = ind1 - offset;//f function location
         char h = compose(x[ind0], x[ind1]);
         x[ind1] = h;//place composed function into g location
      }
   }
}

//sweepdown stage of parallel scan
inline void sweepdown1(__local uint* x, int m) {
   int lid = get_local_id(0);
   int ind1 = (lid*2)+1;//g function location
   int depth = log2(m);
   for(int d=depth-1; d>-1; --d) {
      barrier(CLK_LOCAL_MEM_FENCE);//sync work items
      int mask = (0x1 << d) - 1;
      if((lid & mask) == mask) {//mask unused work items
         int offset = 0x1 << d;
         int ind0 = ind1 - offset;//f function location
         char temp = x[ind1];//store g function
         char h = compose(x[ind0], x[ind1]);
         x[ind1] = h;
         x[ind0] = temp;//place g into f location
      }
   }
}

//kernel to do a parallel scan using COMPOSE
/* data : function array
   x : local array to do computations on
   part : array to hold partial results from work groups
   n : length of data
*/

__kernel void parScanCompose(
   __global char* data, //length n
   __local  char* x,    //length m
   __global char* part, //length k
            uint n) {

   int wx = get_local_size(0);
   //global identifiers
   int gid = get_global_id(0);
   int index0 = (gid*2);
   int index1 = (gid*2)+1;
   //local identifiers
   int lid = get_local_id(0);
   int local_index0 = (lid*2);
   int local_index1 = (lid*2)+1;
   int grpid = get_group_id(0);
   //list lengths
   int m = wx*2;
   int k = get_num_groups(0);
   //copy data into local memory
   //pad local data to make local array powers of two
   x[local_index0] = (index0 < n) ? data[index0] : 2;
   x[local_index1] = (index1 < n) ? data[index1] : 2;
   //store initial data to make scan inclusive
   char2 initial_data = (char2)(x[local_index0], x[local_index1]);

   //sweepup on each subarray
   sweepup1(x, m);
   //last workitem puts the identity into the end of the array
   //save partial result from each work group
   if (lid == (wx-1)) {
      part[grpid] = x[local_index1];
      x[local_index1] = 2;//10 i.e. 2 is the identity
   }
   //sweepdown on each subarray
   sweepdown1(x,m);

   //compose intial data with calculated data
   char h1 = compose(x[local_index0], initial_data.x);
   char h2 = compose(x[local_index1], initial_data.y);
   //save into local memory
   x[local_index0] = h1;
   x[local_index1] = h2;
   //copy back to global data
   if(index0 < n) {
      data[index0] = x[local_index0];
   }
   if(index1 < n) {
      data[index1] = x[local_index1];
   }

}

//function to do a sweepup for parallel scan COMPOSE
//used on partial results for first scan
//only uses global memory
inline void sweepup2(__global uint* x, int k) {
   int gid = get_global_id(0);
   if(gid<k/2) {//only use work items with relevant data
      int ind1 = (gid*2)+1;//g function location
      int depth = log2(k);
      for(int d=0; d<depth; ++d) {
         barrier(CLK_GLOBAL_MEM_FENCE);//sync all work items
         int mask = (0x1 << d) - 1;
         if((gid & mask) == mask) {//mask unused work items
            int offset = 0x1 << d;
            int ind0 = ind1 - offset;//f function location
            char h = compose(x[ind0], x[ind1]);
            x[ind1] = h1;
         }
      }
   }
}

//function to do a sweepdown for parallel scan COMPOSE
//used on partial results for first scan
//only uses global memory
inline void sweepdown2(__global uint* x, int k) {
   int gid = get_global_id(0);
   if(gid<k/2) {//only use work items with relevant data
      int ind1 = (gid*2)+1;//g function location
      int depth = log2(k);
      for(int d=depth-1; d>-1; --d) {
         barrier(CLK_GLOBAL_MEM_FENCE);//sync all work items
         int mask = (0x1 << d) - 1;
         if((gid & mask) == mask) {//mask out unused work items
            int offset = (0x1 << d)*2;
            int ind0 = ind1 - offset;//f function location
            char temp = x[ind1];//store g function
            char h = compose(x[ind0], x[ind1])
            x[ind1] = h;//place composed function into g location
            x[ind0] = temp;//place g function into f location
         }
      }
   }
}

//kernel to parallel scan partial results of initial scan
//produces final outputs of a parallel scan COMPOSE
/*
   data : function array partially scanned
   x : local array for computations
   part : array of partial results from first scan
   n : length of data
*/
__kernel void parScanComposeFromSubarrays(
   __global char* data, //length n
   __local  char* x,    //length m
   __global char* part, //length k
            uint n) {

   //global identifiers
   int gid = get_global_id(0);
   int index0 = (gid*2);
   int index1 = (gid*2)+1;

   //list lengths
   int m = wx*2;
   int k = get_num_groups(0);

   //sweepup on partial results   
   sweepup2(part, k);
   //last work item puts the identity into the end of the array
   if(gid==(k/2)-1) {
      part[index1] = 2;
   }
   //sweepdown on partial results
   sweepdown2(part, k);
   

   //local identifiers
   int lid = get_local_id(0);
   int local_index0 = (lid*2);
   int local_index1 = (lid*2)+1;
   int grpid = get_group_id(0);

   //copy data into local memory
   x[local_index0] = (index0 < n) ? data[index0] : 2;
   x[local_index1] = (index1 < n) ? data[index1] : 2;

   //compose current data with result from previous group
   char h1 = compose(part[grpid], x[local_index0]);
   x[local_index0] = h1;

   char h2 = compose(part[grpid], x[local_index1]);
   x[local_index1] = h2;

   //copy back to global data
   if(index0 < n) {
      data[index0] = x[local_index0];
   }
   if(index1 < n) {
      data[index1] = x[local_index1];
   }
}

__kernel void parScanComposeFuncInc(__global char* func, uint size) {
   uint gid = get_global_id(0);
   uint ind1 = (gid*2)+1;
   uint depth = log2(size);
   
   //scan step
   for(uint d=0; d<depth; ++d){
      barrier(CLK_GLOBAL_MEM_FENCE);
      int mask = (0x1 << d) - 1;
      if((gid & mask) == mask) {
         uint offset = 0x1 << d;
         uint ind0 = ind1 - offset;
         char h = compose(func[ind0], func[ind1]);
         func[ind1] = h;
      }
   }

   //post scan inclusive step
   for(uint stride = size/4; stride > 0; stride /= 2){
      barrier(CLK_GLOBAL_MEM_FENCE);
      uint ind = 2*stride*(gid + 1) - 1;
      if(ind + stride < size){
         char h = compose(func[ind], func[ind+stride]);
         func[ind+stride] = h;
      }
   }
}

inline void parScanAdd(__global uint* data, uint size){
   uint gid = get_global_id(0);
   uint ind1 = (gid*2)+1;
   uint depth = log2(size);
   
   //scan step
   for(uint d=0; d<depth; ++d){
      barrier(CLK_GLOBAL_MEM_FENCE);
      int mask = (0x1 << d) - 1;
      if((gid & mask) == mask) {
         uint offset = 0x1 << d;
         uint ind0 = ind1 - offset;
         data[ind1] += data[ind0];
      }
   }

   //post scan inclusive step
   for(uint stride = size/4; stride > 0; stride /= 2){
      barrier(CLK_GLOBAL_MEM_FENCE);
      uint ind1 = 2*stride*(gid + 1) - 1;
      if(ind1 + stride < size){
         data[ind1 + stride] += data[ind1];
      }
   }
}

/* specChars: SEP, OPEN, CLOSE, ESC */

__kernel void initFunc(__global char* S,
       __global char* specChars, __global uint S_length, 
       __global char* escape, __global char* function) {

   uint global_addr = get_global_id(0);
   char input = S[global_addr];
   
   char open = (input==specChars[1]);
   char close = (input==specChars[2]);
   escape[global_addr] = (input==specChars[3]);
   
   function[global_addr] = 0;
   function[global_addr] |= open;
   function[global_addr] |= (!close || escape[global_addr-1] || open) << 1;
}

/* kernel to find the separators in S using the calculated functions */
__kernel void findSep(__global uint* function, uint size,
      __global char* S, __global uint* separator, 
      char SEP, char firstCharacter, __global uint* final_results) {
   //global identifier
   uint gid = get_global_id(0);
   //determine if char at gid is a valid separator
   separator[gid] = (S[gid] == SEP) && !(function[gid] & 1<<firstCharacter);
   //perform a parallel scan - add on the array of valid separators
   
   parScanAdd(separator, size);
   uint scanResult = separator[gid];

   //store locations in final result array
   if(((gid==0) && (S[gid]==SEP)) || (scanResult !=separator[gid-1])) {
      final_results[scanResult-1] = gid;
   }

}
