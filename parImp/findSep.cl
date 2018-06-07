/* specChars: SEP, OPEN, CLOSE, ESC */

__kernel void findSep(__global char* S, 
      __local uint* local_result, __global uint* group_result,
       __global char* specChars, __global uint S_length, 
       __global uint* escape, __global uint* open, __global uint* close,
       __global uint** function, __global uint* delimited,
       __global uint* separator) {

   
   uint global_addr, local_addr;

   uint global_addr = get_global_id(0);
   char input = S[global_addr];

   uint local_addr = get_local_id(0);
   
   open[global_addr] = (input==specChars[1]);
   close[global_addr] = (input==specChars[2]);
   escape[global_addr] = (input==specChars[3]);

   uint func0 = open[global_addr];
   uint func1 = !close[global_addr] || escape[global_addr] || open[global_addr];
   function[global_addr] = {func0, func1};

   parallelScan(function, Compose);

   delimited[global_addr] = function[global_addr][delimited[0]];
   separator[global_addr] = (input==SEP) && !delimited[global_addr];

   parallelScan(separator, +);

   if(separator[global_addr] != separator[global_addr+1]) {
      group_result[separator[global_addr]] = global_addr;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
   }
}
