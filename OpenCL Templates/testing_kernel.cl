inline void parScanAdd(__global uint* data, uint size){
   uint gid = get_global_id(0);
   if(gid<size/2) {
      uint ind1 = (gid*2)+1;
      uint depth = log2((float)size);
      
      //scan step
      for(uint d=0; d<depth; ++d){
         barrier(CLK_GLOBAL_MEM_FENCE);
         int mask = (0x1 << d) - 1;
         if(((gid & mask) == mask) && (ind1 < size)) {
            uint offset = 0x1 << d;
            uint ind0 = ind1 - offset;
            data[ind1] += data[ind0];
         }
      }

      //post scan inclusive step
      for(uint stride = size/4; stride > 0; stride /= 2){
         barrier(CLK_GLOBAL_MEM_FENCE);
         uint ind = 2*stride*(gid + 1) - 1;
         if(ind + stride < size){
            data[ind + stride] += data[ind];
         }
      }
   }
}

__kernel void add_numbers(__global uint* group_result) {

   group_result = {1,2,3,4,5,6,7,8};
   parScanAdd(group_result, 8);
}