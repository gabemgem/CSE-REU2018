inline uint2 compose(uint f0, uint f1, uint g0, uint g1) {
	uint[] f = {f0, f1};
	uint[] g = {g0, g1};
	uint[] h = {0, 0};
	h[0] = g[f[0]];
	h[1] = g[f[1]];
	return (uint2)(h[0], h[1]);

}

inline void scan(__global uint* x, uint size){
    uint gid = get_global_id(0);
    uint ind1 = (gid*4)+2;
    uint depth = log2(n/2); //every two indicies representing 1 entry
    for(uint d=0; d<depth; ++d){
        barrier(CLK_GLOBAL_MEM_FENCE);
        int mask = (0x1 << d) - 1;
		if((gid & mask) == mask) {
			uint offset = (0x1 << d)*2;
			uint ind0 = ind1 - offset;
			uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1);
			x[ind1] = h.x;
			x[ind1+1] = h.y;
		}
    }
}

inline void inclusive_step(__global uint* x, uint size){
    for(uint sub_size = n/2; sub_size > 2; sub_size /= 2){
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        int mask = (sub_size / 2) - 1,
            stride = sub_size / 4;
        
        if((gid & mask) == mask){
            uint ind0 = gid * 2;
            uint ind1 = (gid + stride) * 2;

            uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1+1]);
            x[ind1] = h.x;
            x[ind1+1] = h.y;
        }
    }
}

__kernel void parScanComposeInclusive(__global uint* func, uint size) {
    
    //wrote code in other functions for easier writing

    //scan step
    scan(func, size);

    //post scan inclusive step
    inclusive_step(func, size);

}
