//function to compose 2-variable boolean functions
//functions represented as two bits of a char
//h = g(f)
inline char compose(char f, char g) {
   char[] __f = {f & 1, (f & 2) >> 1};
   char[] __g = {g & 1, (g & 2) >> 1};
   
   char h = 0;

   h |= g[f[0]];
   h |= g[f[1]] << 1;

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
   __global    char* data, //length n
   __local  char* x, //length m
   __global    char* part, //length k
            uint n) {

   int wx = get_local_size(0);
   //global identifiers
   int gid = get_global_id(0);
   int index0 = (gid*4);
   int index1 = (gid*4)+2;
   //local identifiers
   int lid = get_local_id(0);
   int local_index0 = (lid*4);
   int local_index1 = (lid*4)+2;
   int grpid = get_group_id(0)*2;
   //list lengths
   int m = wx*4;
   int k = get_num_groups(0);
   //copy data into local memory
   //pad local data to make local array powers of two
   x[local_index0] = (index0 < n) ? data[index0] : 0;
   x[local_index0+1] = (index0+1 < n) ? data[index0+1] : 1;
   x[local_index1] = (index1 < n) ? data[index1] : 0;
   x[local_index1+1] = (index1+1 < n) ? data[index1+1] : 1;
   //store initial data to make scan inclusive
   uint4 initial_data = (uint4)(x[local_index0], x[local_index0+1], 
                        x[local_index1], x[local_index1+1]);

   //sweepup on each subarray
   sweepup1(x, m);
   //last workitem puts the identity into the end of the array
   //save partial result from each work group
   if (lid == (wx-1)) {
      part[grpid] = x[local_index1];
      part[grpid+1] = x[local_index1+1];
      x[local_index1] = 0;
      x[local_index1+1] = 1;
   }
   //sweepdown on each subarray
   sweepdown1(x,m);

   //compose intial data with calculated data
   uint2 h1 = compose(x[local_index0], x[local_index0+1], 
                  initial_data.x, initial_data.y);
   uint2 h2 = compose(x[local_index1], x[local_index1+1],
                  initial_data.z, initial_data.w);
   //save into local memory
   x[local_index0] = h1.x;
   x[local_index0+1] = h1.y;
   x[local_index1] = h2.x;
   x[local_index1+1] = h2.y;
   //copy back to global data
   if(index0 < n) {
      data[index0] = x[local_index0];
      data[index0+1] = x[local_index0+1];
   }
   if(index1 < n) {
      data[index1] = x[local_index1];
      data[index1+1] = x[local_index1+1];
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
   __global    uint* data, //length n
   __local  uint* x,    //length m
   __global    uint* part, //length k
            uint n) {

   //global identifiers
   int gid = get_global_id(0);
   int index0 = (gid*4);
   int index1 = (gid*4)+2;

   //list lengths
   int m = wx*4;
   int k = get_num_groups(0);

   //sweepup on partial results   
   sweepup2(part, k);
   //last work item puts the identity into the end of the array
   if(gid==(k/2)-1) {
      part[index1] = 0;
      part[index1+1] = 1;
   }
   //sweepdown on partial results
   sweepdown2(part, k);
   

   //local identifiers
   int lid = get_local_id(0);
   int local_index0 = (lid*4);
   int local_index1 = (lid*4)+2;
   int grpid = get_group_id(0)*2;

   //copy data into local memory
   x[local_index0] = (index0 < n) ? data[index0] : 0;
   x[local_index0+1] = (index0+1 < n) ? data[index0+1] : 1;
   x[local_index1] = (index1 < n) ? data[index1] : 0;
   x[local_index1+1] = (index1+1 < n) ? data[index1+1] : 1;

   //compose current data with result from previous group
   uint2 h1 = compose(part[grpid], part[grpid+1], 
      x[local_index0], x[local_index0+1]);
   x[local_index0] = h1.x;
   x[local_index0+1] = h1.y;

   uint2 h2 = compose(part[grpid], part[grpid+1], 
      x[local_index1], x[local_index1+1]);
   x[local_index1] = h2.x;
   x[local_index1+1] = h2.y;

   //copy back to global data
   if(index0 < n) {
      data[index0] = x[local_index0];
      data[index0+1] = x[local_index0+1];
   }
   if(index1 < n) {
      data[index1] = x[local_index1];
      data[index1+1] = x[local_index1+1];
   }
}

inline void scan(__global uint* x, uint size){
    uint gid = get_global_id(0);
    uint ind1 = (gid*2)+1;
    uint depth = log2(size);
    for(uint d=0; d<depth; ++d){
        barrier(CLK_GLOBAL_MEM_FENCE);
        int mask = (0x1 << d) - 1;
		if((gid & mask) == mask) {
			uint offset = 0x1 << d;
			uint ind0 = ind1 - offset;
			char h = compose(x[ind0], x[ind1]);
			x[ind1] = h;
		}
    }
}

inline void inclusive_step(__global char* x, uint size){
    for(uint sub_size = n/2; sub_size > 2; sub_size /= 2){
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        int mask = (sub_size / 2) - 1,
            stride = sub_size / 4;
        
        if((gid & mask) == mask){
            uint ind0 = gid;
            uint ind1 = gid + stride;

            char h = compose(x[ind0], x[ind1]);
            x[ind1] = h;
        }
    }
}

__kernel void parScanComposeInclusive(__global char* func, uint size) {
    //wrote code in other functions for easier writing

    //scan step
    scan(func, size);

    //post scan inclusive step
    inclusive_step(func, size);

}

/* specChars: SEP, OPEN, CLOSE, ESC */

__kernel void calcFunc(__global char* S,
       __global char* specChars, __global uint S_length, 
       __global uint* escape, __global char* function) {

   
   uint global_addr, local_addr;

   uint global_addr = get_global_id(0);
   char input = S[global_addr];
   uint index = global_addr*2;

   uint local_addr = get_local_id(0);
   
   char open = (input==specChars[1]);
   char close = (input==specChars[2]);
   escape[global_addr] = (input==specChars[3]);
   
   function[index] = 0;
   function[index] |= open;
   function[index] |= (!close || escape[global_addr-1] || open) << 1;

   parallelScanCompose(function);

   if(global_addr == 0){
      delimited[global_addr] = open;
   }

   delimited[global_addr] = (function[global_addr] & (delimited[0] + 1)) >> delimited[0];
   separator[global_addr] = (input==SEP) && !delimited[global_addr];

   parallelScan(separator);

   if(separator[global_addr] != separator[global_addr+1]) {
      group_result[separator[global_addr]] = global_addr;
   }

}
