#define SEP ','
#define OPEN '['
#define CLOSE ']'
#define ESC '\\'

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

__kernel void scanStep(__global char* func, uint size, uint d){
   uint gid = get_global_id(0);

   int mask = (0x1 << d) - 1;
   char selected = ((gid & mask) == mask) && (gid < size/2);
   
   uint ind1 = (selected) ? (gid*2)+1 : 0;
   uint offset = 0x1 << d;
   uint ind0 = (selected) ? ind1 - offset : 0;
   
   char h = compose(func[ind0], func[ind1]);
   func[ind1] = (selected) ? h : func[ind1];
}

__kernel void postScanIncStep(__global char* func, uint size, uint stride){
   uint gid = get_global_id(0);

   uint ind = (2*stride*(gid + 1)) - 1;
   uint ind2 = ind + stride;
   char selected = ind2 < size;
   
   ind = (selected) ? ind : 0;
   ind2 = (selected) ? ind2 : 0;

   char h = compose(func[ind], func[ind2]);
   func[ind2] = (selected) ? h : func[ind2];
}

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

__kernel void addPostScanIncStep(__global uint* data, uint size, uint stride){
   uint gid = get_global_id(0);

   uint ind = (2*stride*(gid + 1)) - 1;
   uint ind2 = ind + stride;
   char selected = ind2 < size;
   
   ind = (selected) ? ind : 0;
   ind2 = (selected) ? ind2 : 0;

   uint h = data[ind] + data[ind2];
   data[ind2] = (selected) ? h : data[ind2];
}

__kernel void initFunc(__global char* S, uint S_length, 
       __global char* escape, __global char* function) {

   uint gid = get_global_id(0);
   char input = S[gid];

   char open = (input==OPEN);
   char close = (input==CLOSE);
   escape[gid] = (input==ESC);
   function[gid] = 0;
   function[gid] |= open;
   function[gid] |= ((gid != 0) ? ((!close) || escape[gid-1]) : (!close)) << 1;
}

/* kernel to find the separators in S using the calculated functions */
__kernel void findSep(__global char* function, uint size,
      __global char* S, char firstCharacter,
      __global uint* final_results) {
   //global identifier
   uint gid = get_global_id(0);
   //determine if char at gid is a valid separator
<<<<<<< HEAD
   separator[gid] = (S[gid] == SEP) && !(function[gid] & 1<<firstCharacter);
   //perform a parallel scan - add on the array of valid separators
   
   
   parScanAdd(separator, size);
   //final_results[gid] = separator[gid];
   /*
   //store locations in final result array
   if(((gid==0) && (S[gid]==SEP)) || ((gid!=0) && (separator[gid] != separator[gid-1]))) {
      final_results[separator[gid]-1] = gid;
   }
   */
=======
   final_results[gid] = (S[gid] == SEP) && !(function[gid] & (1<<firstCharacter));
}
>>>>>>> Staging

__kernel void compressResults(__global uint* final_results, 
                              __global uint* locs) {
   uint gid = get_global_id(0);
   uint val = final_results[gid];
   barrier(CLK_GLOBAL_MEM_FENCE);

   if((gid==0 && val!=0) || (gid!=0 && val!=final_results[gid-1])) {
      locs[val-1] = gid;
   }
}
