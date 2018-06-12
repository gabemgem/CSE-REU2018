//function to compose 2-variable boolean functions
//h = g(f)
inline uint2 compose(uint f0, uint f1, uint g0, uint g1) {
   uint[] f = {f0, f1};
   uint[] g = {g0, g1};
   uint[] h = {0, 0};
   h[0] = g[f[0]];
   h[1] = g[f[1]];
   return (uint2)(h[0], h[1]);

}

//sweepup stage of parallel scan
inline void sweepup1(__local uint* x, int m) {
   int lid = get_local_id(0);
   int ind1 = (lid*4)+2;//g function location
   int depth = log2(m/2);
   for(int d=0; d<depth; ++d) {
      barrier(CLK_LOCAL_MEM_FENCE);//sync work items
      int mask = (0x1 << d) - 1;
      if((lid & mask) == mask) {//mask unused work items
         int offset = (0x1 << d)*2;
         int ind0 = ind1 - offset;//f function location
         uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1);
         x[ind1] = h.x;//place composed function into g location
         x[ind1+1] = h.y;
      }
   }
}

//sweepdown stage of parallel scan
inline void sweepdown1(__local uint* x, int m) {
   int lid = get_local_id(0);
   int ind1 = (lid*4)+2;//g function location
   int depth = log2(m/2);
   for(int d=depth-1; d>-1; --d) {
      barrier(CLK_LOCAL_MEM_FENCE);//sync work items
      int mask = (0x1 << d) - 1;
      if((lid & mask) == mask) {//mask unused work items
         int offset = (0x1 << d)*2;
         int ind0 = ind1 - offset;//f function location
         int temp0 = x[ind1];//store g function
         int temp1 = x[ind1+1];
         uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1)
         x[ind1] = h.x;//place composed function into g location
         x[ind1+1] = h.y;
         x[ind0] = temp0;//place g into f location
         x[ind0+1] = temp1;
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
   __global    uint* data, //length n
   __local  uint* x, //length m
   __global    uint* part, //length m
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
      int ind1 = (gid*4)+2;//g function location
      int depth = log2(k/2);
      for(int d=0; d<depth; ++d) {
         barrier(CLK_GLOBAL_MEM_FENCE);//sync all work items
         int mask = (0x1 << d) - 1;
         if((gid & mask) == mask) {//mask unused work items
            int offset = (0x1 << d)*2;
            int ind0 = ind1 - offset;//f function location
            uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1);
            x[ind1] = h.x;//save composed function into g location
            x[ind1+1] = h.y;
         }
      }
   }
}

//function to do a sweepdown for parallel scan COMPOSE
//used on partial results for first scan
//only uses global memory
inline void sweepdown2(__global uint* x, int k) {
   int gid = get_global_id(0);
   if(gid<k/2) {
      int ind1 = (gid*4)+2;
      int depth = log2(k/2);
      for(int d=depth-1; d>-1; --d) {
         barrier(CLK_GLOBAL_MEM_FENCE);
         int mask = (0x1 << d) - 1;
         if((gid & mask) == mask) {
            int offset = (0x1 << d)*2;
            int ind0 = ind1 - offset;
            int temp0 = x[ind1];
            int temp1 = x[ind1+1];
            uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1)
            x[ind1] = h.x;
            x[ind1+1] = h.y;
            x[ind0] = temp0;
            x[ind0+1] = temp1;
         }
      }
   }
}

__kernel void parScanComposeFromSubarrays(
   __global    uint* data, //length n
   __local  uint* x,    //length m
   __global    uint* part, //length m
            uint n) {
   int gid = get_global_id(0);
   int index0 = (gid*4);
   int index1 = (gid*4)+2;

   
   //list lengths
   int m = wx*4;
   int k = get_num_groups(0);

   
   sweepup2(part, k);
   if(gid==k-1) {
      part[index1] = 0;
      part[index1+1] = 1;
   }
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



/* specChars: SEP, OPEN, CLOSE, ESC */

__kernel void calcFunc(__global char* S,
       __global char* specChars, __global uint S_length, 
       __global uint* escape, __global uint* function) {

   
   uint global_addr, local_addr;

   uint global_addr = get_global_id(0);
   char input = S[global_addr];
   uint index = global_addr*2;

   uint local_addr = get_local_id(0);
   
   char open = (input==specChars[1]);
   char close = (input==specChars[2]);
   escape[global_addr] = (input==specChars[3]);

   function[index] = open;
   function[(index)+1] = !close || 
                         escape[global_addr-1] || open;

   parallelScanCompose(function);

   delimited[global_addr] = function[global_addr][delimited[0]];
   separator[global_addr] = (input==SEP) && !delimited[global_addr];

   parallelScan(separator);

   if(separator[global_addr] != separator[global_addr+1]) {
      group_result[separator[global_addr]] = global_addr;
   }

}
