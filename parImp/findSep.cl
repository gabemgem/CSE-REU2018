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
