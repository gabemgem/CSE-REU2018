/* specChars: SEP, OPEN, CLOSE, ESC */

__kernel void findSep(__global char* S,  __global uint* group_result,
       __global char* specChars, __global uint S_length, 
       __global uint* escape, __global uint** function,
       __global uint* delimited, __global uint* separator) {

   
   uint global_addr, local_addr;

   uint global_addr = get_global_id(0);
   char input = S[global_addr];

   uint local_addr = get_local_id(0);
   
   char open = (input==specChars[1]);
   char close = (input==specChars[2]);
   escape[global_addr] = (input==specChars[3]);

   function[global_addr*2] = open;
   function[(global_addr*2)+1] = !close || 
                                 escape[global_addr-1] || open;

   parallelScan(function, Compose);

   delimited[global_addr] = function[global_addr][delimited[0]];
   separator[global_addr] = (input==SEP) && !delimited[global_addr];

   parallelScan(separator, +);

   if(separator[global_addr] != separator[global_addr+1]) {
      group_result[separator[global_addr]] = global_addr;
   }

}
