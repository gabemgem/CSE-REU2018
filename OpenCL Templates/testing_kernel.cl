inline void parScanAdd(__global uint* data, uint size){
   uint gid = get_global_id(0);
   uint ind1 = (gid*2)+1;
   uint depth = (uint)log2((float)size);
   
   //scan step
   for(uint d=0; d<depth; ++d){
      barrier(CLK_GLOBAL_MEM_FENCE);
      int mask = (0x1 << d) - 1;
      if((gid<size/2) && ((gid & mask) == mask)) {
         uint offset = 0x1 << d;
         uint ind0 = ind1 - offset;
         data[ind1] += data[ind0];
      }
   }
   
   //post scan inclusive step
   for(int stride = size/4; stride > 0; stride /= 2){
      barrier(CLK_GLOBAL_MEM_FENCE);
      int ind = 2*stride*(gid + 1) - 1;
      if((gid<size/2) && (ind + stride < size)){
         data[ind + stride] += data[ind];
      }
      
   }
      
   
}

__kernel void add_numbers(__global uint* group_result) {
   uint gid = get_global_id(0);
   group_result[gid] = gid % 3 == 2;
   parScanAdd(group_result, 32);
}