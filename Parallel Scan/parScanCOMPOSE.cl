

inline void sweepup1(__local uint* x, int m) {
	int lid = get_local_id(0);
	int ind1 = (lid*4)+2;
	int depth = log2(m)/2;
	for(int d=0; d<depth; ++d) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int mask = (0x1 << d) - 1;
		if((lid & mask) == mask) {
			int offset = (0x1 << d)*2;
			int ind0 = ind1 - offset;
			uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1);
			x[ind1] = h.x;
			x[ind1+1] = h.y;
		}
	}
}

inline void sweepdown1(__local uint* x, int m) {
	int lid = get_local_id(0);
	int ind1 = (lid*4)+2;
	int depth = log2(m)/2;
	for(int d=depth-1; d>-1; --d) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int mask = (0x1 << d) - 1;
		if((lid & mask) == mask) {
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

inline uint2 compose(uint f0, uint f1, uint g0, uint g1) {
	uint* f = {f0, f1};
	uint* g = {g0, g1};
	uint* h = {0, 0};
	h[0] = g[f[0]];
	h[1] = g[f[1]];
	return (uint2)(h[0], h[1]);

}

__kernel void parScanCompose(
	__global 	uint* data,	//length n
	__local 	uint* x,	//length m
	__global 	uint* part,	//length m
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
	x[local_index0] = (index0 < n) ? data[index0] : 0;
	x[local_index0+1] = (index0+1 < n) ? data[index0+1] : 1;
	x[local_index1] = (index1 < n) ? data[index1] : 0;
	x[local_index1+1] = (index1+1 < n) ? data[index1+1] : 1;

	//sweepup on each subarray
	sweepup1(x, m);
	//last workitem puts the identity into the end of the array
	if (lid == (wx-1)) {
		part[grpid] = x[local_index1];
		part[grpid+1] = x[local_index1+1];
		x[local_index1] = 0;
		x[local_index1+1] = 1;
	}
	//sweepdown on each subarray
	sweepdown1(x,m);


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


inline void sweepup2(__global uint* x, int k) {
	int gid = get_global_id(0);
	if(gid<k/2) {
		int ind1 = (gid*4)+2;
		int depth = log2(k/2);
		for(int d=0; d<depth; ++d) {
			barrier(CLK_GLOBAL_MEM_FENCE);
			int mask = (0x1 << d) - 1;
			if((gid & mask) == mask) {
				int offset = (0x1 << d)*2;
				int ind0 = ind1 - offset;
				uint2 h = compose(x[ind0], x[ind0+1], x[ind1], x[ind1]+1);
				x[ind1] = h.x;
				x[ind1+1] = h.y;
			}
		}
	}
}

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
	__global 	uint* data,	//length n
	__local 	uint* x, 	//length m
	__global 	uint* part,	//length m
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
