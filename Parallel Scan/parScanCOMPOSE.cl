

inline void sweepup(__local uint* x, int m) {
	int lid = get_local_id(0);
	int ind1 = (lid*2)+1;
	int depth = log2(m);
	for(int d=0; d<depth; ++d) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int mask = (0x1 << d) - 1;
		if((lid & mask) == mask) {
			int offset = (0x1 << d);
			int ind0 = ind1 - offset;
			x[ind1] += x[ind0];
		}
	}
}

inline void sweepdown(__local uint* x, int m) {
	int lid = get_local_id(0);
	int ind1 = (lid*2)+1;
	int depth = log2(m);
	for(int d=depth-1; d>-1; --d) {
		barrier(CLK_LOCAL_MEM_FENCE);
		int mask = (0x1 << d) - 1;
		if((lid & mask) == mask) {
			int offset = (0x1 << d);
			int ind0 = ind1 - offset;
			int temp = x[ind1];
			x[ind1] += x[ind0];
			x[ind0] = temp;
		}
	}
}

inline 

__kernel void parScanCompose(
	__global 	uint* data,	//length n
	__local 	uint* x,	//length m
	__global 	uint* part,	//length m
				uint n) {

	int wx = get_local_size(0);
	//global identifiers
	int gid = get_global_id(0);
	int index0 = (gid*2);
	int index1 = (gid*2)+1;
	//local identifiers
	int lid = get_local_id(0);
	int local_index0 = (lid*2);
	int local_index1 = (lid*2)+1;
	int grpid = get_group_id(0);
	//list lengths
	int m = wx*2;
	int k = get_num_groups(0);
	//copy data into local memory
	x[local_index0] = (index0 < n) ? data[index0] : 0;
	x[local_index1] = (index1 < n) ? data[index1] : 0;

	//sweepup on each subarray
	sweepup(x, m);
	//last workitem saves last element of each subarray and zeroes
	if (lid == (wx-1)) {
		part[grpid] = x[local_index1];
		x[local_index1] = 0;
	}
	//sweepdown on each subarray
	sweepdown(x,m);

	//copy back to global data
	if(index0 < n) {
		data[index0] = x[local_index0];
	}
	if(index1 < n) {
		data[index1] = x[local_index1];
	}

}