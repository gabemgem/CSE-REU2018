__kernel void initFunc(__global uint * vals, uint size){
   uint gid = get_global_id(0);
   vals[gid] = (gid * 2) + 1;
}