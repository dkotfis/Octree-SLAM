/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * 
 * 
 * 
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Bottom-level digit-reduction/counting kernel
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {




/******************************************************************************
 * Cycle-processing Routines
 ******************************************************************************/

template <int BYTE>
__device__ __forceinline__ int DecodeInt(int encoded){
	
	int retval;
	ExtractKeyBits<int, BYTE * 8, 8>::Extract(retval, encoded);
	return retval;
}


//-----------------------------------------------------------------------------

template <int PARTIAL>
__device__ __forceinline__  void ReduceLanePartial(
	int local_counts[4], 
	int *scan_lanes, 
	int lane_offset) 
{
	unsigned char* encoded = (unsigned char *) &scan_lanes[lane_offset + (PARTIAL * B40C_WARP_THREADS)];
	local_counts[0] += encoded[0];
	local_counts[1] += encoded[1];
	local_counts[2] += encoded[2];
	local_counts[3] += encoded[3];
}

template <int LANE, int REDUCTION_LANES, int REDUCTION_LANES_PER_WARP, int REDUCTION_PARTIALS_PER_LANE, int LANE_PARTIALS_PER_THREAD>
__device__ __forceinline__  void ReduceLanePartials(
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int *scan_lanes, 
	int lane_offset) 
{		
	lane_offset += (LANE * REDUCTION_PARTIALS_PER_LANE * B40C_RADIXSORT_WARPS);
	if ((B40C_RADIXSORT_WARPS < REDUCTION_LANES) || (lane_offset < REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE)) {
		if (LANE_PARTIALS_PER_THREAD > 0) ReduceLanePartial<0>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 1) ReduceLanePartial<1>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 2) ReduceLanePartial<2>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 3) ReduceLanePartial<3>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 4) ReduceLanePartial<4>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 5) ReduceLanePartial<5>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 6) ReduceLanePartial<6>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 7) ReduceLanePartial<7>(local_counts[LANE], scan_lanes, lane_offset);
	}
}


template <
	int REDUCTION_LANES, 
	int REDUCTION_LANES_PER_WARP, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE> 
__device__ __forceinline__ void ReduceEncodedCounts(
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int *scan_lanes,
	int warp_id,
	int warp_idx)
{
	const int LANE_PARTIALS_PER_THREAD = REDUCTION_PARTIALS_PER_LANE / B40C_WARP_THREADS;
	SuppressUnusedConstantWarning(LANE_PARTIALS_PER_THREAD);
	
	int lane_offset = (warp_id << LOG_REDUCTION_PARTIALS_PER_LANE) + warp_idx;	// my warp's (first-lane) reduction offset

	if (REDUCTION_LANES_PER_WARP > 0) ReduceLanePartials<0, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 1) ReduceLanePartials<1, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 2) ReduceLanePartials<2, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 3) ReduceLanePartials<3, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
}
	

template <typename K, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
__device__ __forceinline__ void Bucket(
	K key, 
	int *encoded_reduction_col,
	PreprocessFunctor preprocess = PreprocessFunctor()) 
{
	preprocess(key);

	int lane;
	ExtractKeyBits<K, BIT + 2, RADIX_BITS - 2>::Extract(lane, key);

	if (B40C_FERMI(__CUDA_ARCH__)) {	
	
		// GF100+ has special bit-extraction instructions (instead of shift+mask)
		int quad_byte;
		if (RADIX_BITS < 2) { 
			ExtractKeyBits<K, BIT, 1>::Extract(quad_byte, key);
		} else {
			ExtractKeyBits<K, BIT, 2>::Extract(quad_byte, key);
		}
		unsigned char *encoded_col = (unsigned char *) &encoded_reduction_col[FastMul(lane, REDUCTION_PARTIALS_PER_LANE)];
		encoded_col[quad_byte]++;

	} else {

		// GT200 can save an instruction because it can source an operand 
		// directly from smem
		const int BYTE_ENCODE_SHIFT 		= 0x3;
		const K QUAD_MASK 					= (RADIX_BITS < 2) ? 0x1 : 0x3;
		int quad_shift = MagnitudeShift<K, BYTE_ENCODE_SHIFT - BIT>(key & (QUAD_MASK << BIT));
		encoded_reduction_col[FastMul(lane, REDUCTION_PARTIALS_PER_LANE)] += (1 << quad_shift);
	}
}


template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor, int CYCLES>
struct LoadOp;

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		K key;
		GlobalLoad<K, CACHE_MODIFIER >::Ld(key, d_in_keys, block_offset);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(key, encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 1), encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 2), encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		K keys[8];
			
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[0], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[1], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 1));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[2], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 2));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[3], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 3));

		if (B40C_FERMI(__CUDA_ARCH__)) __syncthreads();
		
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[4], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 4));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[5], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 5));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[6], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 6));
		GlobalLoad<K, CACHE_MODIFIER >::Ld(keys[7], d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 7));
		
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[0], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[1], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[2], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[3], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[4], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[5], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[6], encoded_reduction_col);
		Bucket<K, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[7], encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 8), encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 16), encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 32), encoded_reduction_col);
	}
};

template <typename K, CacheModifier CACHE_MODIFIER, int RADIX_BITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 64), encoded_reduction_col);
	}
};



template <int REDUCTION_LANES>
__device__ __forceinline__ void ResetEncodedCarry(
	int *encoded_reduction_col)
{
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) REDUCTION_LANES; SCAN_LANE++) {
		encoded_reduction_col[SCAN_LANE * B40C_RADIXSORT_THREADS] = 0;
	}
}


template <bool UNROLL, typename K, CacheModifier CACHE_MODIFIER, int BIT, int RADIX_BITS, int REDUCTION_LANES, int REDUCTION_LANES_PER_WARP, int LOG_REDUCTION_PARTIALS_PER_LANE, int REDUCTION_PARTIALS_PER_LANE, typename PreprocessFunctor>
struct UnrolledLoads;

// Minimal unrolling
template <typename K, CacheModifier CACHE_MODIFIER, int BIT, int RADIX_BITS, int REDUCTION_LANES, int REDUCTION_LANES_PER_WARP, int LOG_REDUCTION_PARTIALS_PER_LANE, int REDUCTION_PARTIALS_PER_LANE, typename PreprocessFunctor>
struct UnrolledLoads <false, K, CACHE_MODIFIER, BIT, RADIX_BITS, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, PreprocessFunctor>
{
	__device__ __forceinline__ static void Unroll(
		K*			d_in_keys,
		int 		&block_offset,
		int* 		encoded_reduction_col,
		int*		scan_lanes,
		const int&	out_of_bounds,
		int local_counts[REDUCTION_LANES_PER_WARP][4],
		int warp_id,
		int warp_idx) 
	{
		// Unroll batches of loads with occasional reduction to avoid overflow
		while (block_offset + (B40C_RADIXSORT_THREADS * 32) < out_of_bounds) {
		
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 32;
	
			__syncthreads();
	
			// Aggregate back into local_count registers to prevent overflow
			ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE>(
					local_counts, 
					scan_lanes,
					warp_id,
					warp_idx);
	
			__syncthreads();
			
			// Reset encoded counters
			ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
		} 
	}
};

// Unrolled
template <typename K, CacheModifier CACHE_MODIFIER, int BIT, int RADIX_BITS, int REDUCTION_LANES, int REDUCTION_LANES_PER_WARP, int LOG_REDUCTION_PARTIALS_PER_LANE, int REDUCTION_PARTIALS_PER_LANE, typename PreprocessFunctor>
struct UnrolledLoads <true, K, CACHE_MODIFIER, BIT, RADIX_BITS, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, PreprocessFunctor>
{
	__device__ __forceinline__ static void Unroll(
		K*			d_in_keys,
		int 		&block_offset,
		int* 		encoded_reduction_col,
		int*		scan_lanes,
		const int&	out_of_bounds,
		int local_counts[REDUCTION_LANES_PER_WARP][4],
		int warp_id,
		int warp_idx) 
	{
		// Unroll batches of loads with occasional reduction to avoid overflow
		while (block_offset + (B40C_RADIXSORT_THREADS * 128) < out_of_bounds) {
		
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 128;
	
			__syncthreads();
	
			// Aggregate back into local_count registers to prevent overflow
			ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE>(
					local_counts, 
					scan_lanes,
					warp_id,
					warp_idx);
	
			__syncthreads();
			
			// Reset encoded counters
			ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
		} 
		
		if (block_offset + (B40C_RADIXSORT_THREADS * 64) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 64;
		}
		if (block_offset + (B40C_RADIXSORT_THREADS * 32) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 32;
		}
		if (block_offset + (B40C_RADIXSORT_THREADS * 16) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 16;
		}
		if (block_offset + (B40C_RADIXSORT_THREADS * 8) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 8;
		}
		if (block_offset + (B40C_RADIXSORT_THREADS * 4) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 4;
		}
		if (block_offset + (B40C_RADIXSORT_THREADS * 2) < out_of_bounds) {
			LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
			block_offset += B40C_RADIXSORT_THREADS * 2;
		}
	}
};


template <
	typename K, 
	CacheModifier CACHE_MODIFIER,
	int BIT, 
	int RADIX_BITS,
	int RADIX_DIGITS, 
	int REDUCTION_LANES, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE, 
	typename PreprocessFunctor,
	bool UNROLL>
__device__ __forceinline__ void ReductionPass(
	K*			d_in_keys,
	int* 		d_spine,
	int 		block_offset,
	int* 		encoded_reduction_col,
	int*		scan_lanes,
	const int&	out_of_bounds)
{
	const int REDUCTION_LANES_PER_WARP 			= (REDUCTION_LANES > B40C_RADIXSORT_WARPS) ? REDUCTION_LANES / B40C_RADIXSORT_WARPS : 1;	// Always at least one fours group per warp
	const int PARTIALS_PER_ROW = B40C_WARP_THREADS;
	const int PADDED_PARTIALS_PER_ROW = PARTIALS_PER_ROW + 1;

	int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS;
	int warp_idx = threadIdx.x & (B40C_WARP_THREADS - 1);
	
	block_offset += threadIdx.x;
	
	// Each thread is responsible for aggregating an unencoded segment of a fours-group
	int local_counts[REDUCTION_LANES_PER_WARP][4];								
	
	// Initialize local counts
	#pragma unroll 
	for (int LANE = 0; LANE < (int) REDUCTION_LANES_PER_WARP; LANE++) {
		local_counts[LANE][0] = 0;
		local_counts[LANE][1] = 0;
		local_counts[LANE][2] = 0;
		local_counts[LANE][3] = 0;
	}
	
	// Reset encoded counters
	ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);

	// Process loads in bulk (if applicable)
	UnrolledLoads<UNROLL, K, CACHE_MODIFIER, BIT, RADIX_BITS, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, PreprocessFunctor>::Unroll(
		d_in_keys,
		block_offset,
		encoded_reduction_col,
		scan_lanes,
		out_of_bounds + threadIdx.x,
		local_counts, 
		warp_id,
		warp_idx); 
	
	// Process (potentially-partial) loads singly
	while (block_offset < out_of_bounds) {
		LoadOp<K, CACHE_MODIFIER, RADIX_BITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
		block_offset += B40C_RADIXSORT_THREADS;
	}
	
	__syncthreads();

	// Aggregate back into local_count registers 
	ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE>(
		local_counts, 
		scan_lanes,
		warp_id,
		warp_idx);
	
	__syncthreads();
		
	//
	// Reduce the local_counts within each reduction lane within each warp  
	//
	
	// Place into smem
	int lane_base = FastMul(warp_id, PADDED_PARTIALS_PER_ROW * B40C_RADIXSORT_WARPS);	// my warp's (first) reduction lane
	
	#pragma unroll
	for (int i = 0; i < (int) REDUCTION_LANES_PER_WARP; i++) {

		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 0)] = local_counts[i][0];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 1)] = local_counts[i][1];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 2)] = local_counts[i][2];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 3)] = local_counts[i][3];
		
		lane_base += PADDED_PARTIALS_PER_ROW * B40C_RADIXSORT_WARPS;
	}

	__syncthreads();

	// Rake-reduce and write out the digit_count reductions 
	if (threadIdx.x < RADIX_DIGITS) {

		int lane_base = FastMul(threadIdx.x, PADDED_PARTIALS_PER_ROW);
		int digit_count = SerialReduce<PARTIALS_PER_ROW>(scan_lanes + lane_base);

		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		d_spine[spine_digit_offset] = digit_count;
	}
}




template <typename K, typename V, int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_REDUCE_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void LsbRakingReductionKernel(
	int *d_selectors,
	int *d_spine,
	K *d_in_keys,
	K *d_out_keys,
	CtaDecomposition work_decomposition)
{
	const int RADIX_DIGITS 						= 1 << RADIX_BITS;
	const int TILE_ELEMENTS						= B40C_RADIXSORT_TILE_ELEMENTS(__CUDA_ARCH__, K, V);

	const int LOG_REDUCTION_PARTIALS_PER_LANE	= B40C_RADIXSORT_LOG_THREADS;
	const int REDUCTION_PARTIALS_PER_LANE 		= 1 << LOG_REDUCTION_PARTIALS_PER_LANE;

	const int LOG_REDUCTION_LANES 				= (RADIX_BITS >= 2) ? RADIX_BITS - 2 : 0;	// Always at least one fours group
	const int REDUCTION_LANES 					= 1 << LOG_REDUCTION_LANES;

	SuppressUnusedConstantWarning(RADIX_DIGITS);
	
	
	// Each thread gets its own column of fours-groups (for conflict-free updates)
	__shared__ int scan_lanes[REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE];	
	
	int *encoded_reduction_col = &scan_lanes[threadIdx.x];	// first element of column

	// Determine where to read our input
	int selector = (PASS == 0) ? 0 : d_selectors[PASS & 0x1];
	if (selector) d_in_keys = d_out_keys;
	
	// Calculate our threadblock's range
	int block_offset, block_elements;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		block_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		block_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * TILE_ELEMENTS);
		block_elements = work_decomposition.normal_block_elements;
	}
	int out_of_bounds = block_offset + block_elements; 
	if (blockIdx.x == gridDim.x - 1) {
		if (work_decomposition.extra_elements_last_block > 0) {
			out_of_bounds -= TILE_ELEMENTS;
		}
		out_of_bounds += work_decomposition.extra_elements_last_block;
	}
	
	// Perform reduction pass
	ReductionPass<K, NONE, BIT, RADIX_BITS, RADIX_DIGITS, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, PreprocessFunctor, true>(
		d_in_keys,
		d_spine,
		block_offset,
		encoded_reduction_col,
		scan_lanes,
		out_of_bounds);
} 

 

} // namespace b40c

