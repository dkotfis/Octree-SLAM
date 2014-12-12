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
 * Scan-scatter kernel.  The third kernel in a radix-sorting digit-place pass.
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {

/**
 * Register-saving variable qualifier. Can be used when declaring 
 * variables that would otherwise have the same value for all threads in the CTA. 
 */
#if __CUDA_ARCH__ >= 200
	#define _B40C_SCANSCATTER_REG_MISER_ __shared__
#else
	#define _B40C_SCANSCATTER_REG_MISER_ 
#endif

/******************************************************************************
 * Appropriate key-substitutes for use by threads that would otherwise index 
 * past the end of valid global data  
 ******************************************************************************/

template <typename T> 
__device__ __forceinline__ void DefaultExtraValue(T &val) {
}

// Accomodate bizarre introduction of "signed" for char loads
__device__ __forceinline__ void DefaultExtraValue(char &val) {
}

template <> 
__device__ __forceinline__ void DefaultExtraValue<unsigned char>(unsigned char &val) {
	val = (unsigned char) -1;
}

template <> 
__device__ __forceinline__ void DefaultExtraValue<unsigned short>(unsigned short &val) {
	val = (unsigned short) -1;
}

template <> 
__device__ __forceinline__ void DefaultExtraValue<unsigned int>(unsigned int &val) {
	val = (unsigned int) -1;
}

template <> 
__device__ __forceinline__ void DefaultExtraValue<unsigned long>(unsigned long &val) {
	val = (unsigned long) -1;
}

template <> 
__device__ __forceinline__ void DefaultExtraValue<unsigned long long>(unsigned long long &val) {
	val = (unsigned long long) -1;
}

/******************************************************************************
 * Tile-processing Routines
 ******************************************************************************/


template <typename K, int RADIX_BITS, int BIT>
__device__ __forceinline__ int DecodeDigit(K key) 
{
	int retval;
	ExtractKeyBits<K, BIT, RADIX_BITS>::Extract(retval, key);
	return retval;
}


template <typename K, int RADIX_BITS, int BIT, int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	int &digit, 
	int &flag_offset,		// in bytes
	const int LOAD_OFFSET)
{
	const int PADDED_BYTES_PER_LANE 	= PADDED_PARTIALS_PER_LANE * 4;
	const int LOAD_OFFSET_BYTES 		= LOAD_OFFSET * 4;
	const K QUAD_MASK 					= (RADIX_BITS < 2) ? 0x1 : 0x3;
	
	digit = DecodeDigit<K, RADIX_BITS, BIT>(key);
	int lane = digit >> 2;
	int quad_byte = digit & QUAD_MASK;

	flag_offset = LOAD_OFFSET_BYTES + FastMul(lane, PADDED_BYTES_PER_LANE) + quad_byte;
}


template <typename K, int RADIX_BITS, int BIT, int LOADS_PER_CYCLE, int SCAN_LANES_PER_LOAD, int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigits(
	typename VecType<K, 2>::Type keypairs[LOADS_PER_CYCLE],
	int2 digits[LOADS_PER_CYCLE],
	int2 flag_offsets[LOADS_PER_CYCLE])		// in bytes 
{
	if (LOADS_PER_CYCLE > 0) {
		const int LOAD = 0;
		const int LOAD_OFFSET = LOAD * SCAN_LANES_PER_LOAD * PADDED_PARTIALS_PER_LANE;
		DecodeDigit<K, RADIX_BITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[LOAD].x, digits[LOAD].x, flag_offsets[LOAD].x, LOAD_OFFSET);
		DecodeDigit<K, RADIX_BITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[LOAD].y, digits[LOAD].y, flag_offsets[LOAD].y, LOAD_OFFSET);
	}
	if (LOADS_PER_CYCLE > 1) {
		const int LOAD = 1;
		const int LOAD_OFFSET = LOAD * SCAN_LANES_PER_LOAD * PADDED_PARTIALS_PER_LANE;
		DecodeDigit<K, RADIX_BITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[LOAD].x, digits[LOAD].x, flag_offsets[LOAD].x, LOAD_OFFSET);
		DecodeDigit<K, RADIX_BITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[LOAD].y, digits[LOAD].y, flag_offsets[LOAD].y, LOAD_OFFSET);
	}
}


template <typename T, CacheModifier CACHE_MODIFIER, typename PreprocessFunctor>
__device__ __forceinline__ void GuardedLoad(
	T *in, 
	typename VecType<T, 2>::Type &pair,
	int offset,
	const int &extra_elements,
	PreprocessFunctor preprocess = PreprocessFunctor())				
{
	
	if (offset - extra_elements < 0) {
		GlobalLoad<T, CACHE_MODIFIER>::Ld(pair.x, in, offset);
		preprocess(pair.x);
	} else {
		DefaultExtraValue(pair.x);
	}
	
	if (offset + 1 - extra_elements < 0) {
		GlobalLoad<T, CACHE_MODIFIER>::Ld(pair.y, in, offset + 1);
		preprocess(pair.y);
	} else {
		DefaultExtraValue(pair.y);
	}
}


template <typename T, CacheModifier CACHE_MODIFIER, bool UNGUARDED_IO, int LOADS_PER_CYCLE, typename PreprocessFunctor>
struct ReadCycle
{
	__device__ __forceinline__ static void Read(
		typename VecType<T, 2>::Type *d_in, 
		typename VecType<T, 2>::Type pairs[LOADS_PER_CYCLE],
		const int BASE2,
		const int &extra_elements,
		PreprocessFunctor preprocess = PreprocessFunctor())				
	{
		if (UNGUARDED_IO) {

			// N.B. -- I wish we could do some pragma unrolling here too, but we can't with asm statements inside
			if (LOADS_PER_CYCLE > 0) GlobalLoad<typename VecType<T, 2>::Type, CACHE_MODIFIER>::Ld(
					pairs[0], d_in, threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 0));
			if (LOADS_PER_CYCLE > 1) GlobalLoad<typename VecType<T, 2>::Type, CACHE_MODIFIER>::Ld(
					pairs[1], d_in, threadIdx.x + BASE2 + (B40C_RADIXSORT_THREADS * 1));

			#pragma unroll 
			for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
				preprocess(pairs[LOAD].x);
				preprocess(pairs[LOAD].y);
			}
			
		} else {

			// N.B. -- I wish we could do some pragma unrolling here too, but we can't with asm statements inside
			if (LOADS_PER_CYCLE > 0) GuardedLoad<T, CACHE_MODIFIER, PreprocessFunctor>(
					(T*) d_in, pairs[0], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 0), extra_elements);
			if (LOADS_PER_CYCLE > 1) GuardedLoad<T, CACHE_MODIFIER, PreprocessFunctor>(
					(T*) d_in, pairs[1], (threadIdx.x << 1) + (BASE2 << 1) + (B40C_RADIXSORT_THREADS * 2 * 1), extra_elements);
		}
	}
};


template <CacheModifier CACHE_MODIFIER, bool UNGUARDED_IO, int LOADS_PER_CYCLE, typename PreprocessFunctor>
struct ReadCycle<KeysOnlyType, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, PreprocessFunctor>
{
	__device__ __forceinline__ static void Read(
		typename VecType<KeysOnlyType, 2>::Type *d_in, 
		typename VecType<KeysOnlyType, 2>::Type pairs[LOADS_PER_CYCLE],
		const int BASE2,
		const int &extra_elements,
		PreprocessFunctor preprocess = PreprocessFunctor())				
	{
		// Do nothing for KeysOnlyType
	}
};


template <int LOADS_PER_CYCLE>
__device__ __forceinline__ void PlacePartials(
	unsigned char * base_partial,
	int2 digits[LOADS_PER_CYCLE],
	int2 flag_offsets[LOADS_PER_CYCLE]) 
{
	#pragma unroll
	for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
		base_partial[flag_offsets[LOAD].x] = 1;
		base_partial[flag_offsets[LOAD].y] = 1 + (digits[LOAD].x == digits[LOAD].y);
	}
}


template <int LOADS_PER_CYCLE>
__device__ __forceinline__ void ExtractRanks(
	unsigned char * base_partial,
	int2 digits[LOADS_PER_CYCLE],
	int2 flag_offsets[LOADS_PER_CYCLE],
	int2 ranks[LOADS_PER_CYCLE]) 
{
	#pragma unroll
	for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
		ranks[LOAD].x = base_partial[flag_offsets[LOAD].x];
		ranks[LOAD].y = base_partial[flag_offsets[LOAD].y] + (digits[LOAD].x == digits[LOAD].y);
	}
}


template <int RADIX_DIGITS, int LOADS_PER_CYCLE>
__device__ __forceinline__ void UpdateRanks(
	int2 digits[LOADS_PER_CYCLE],
	int2 ranks[LOADS_PER_CYCLE],
	int digit_counts[LOADS_PER_CYCLE][RADIX_DIGITS])
{
	#pragma unroll
	for (int LOAD = 0; LOAD < LOADS_PER_CYCLE; LOAD++) {
		ranks[LOAD].x += digit_counts[LOAD][digits[LOAD].x];
		ranks[LOAD].y += digit_counts[LOAD][digits[LOAD].y]; 
	}
}

template <int RADIX_DIGITS, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void UpdateRanks(
	int2 digits[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS])
{
	#pragma unroll
	for (int CYCLE = 0; CYCLE < CYCLES_PER_TILE; CYCLE++) {
		UpdateRanks<RADIX_DIGITS, LOADS_PER_CYCLE>(digits[CYCLE], ranks[CYCLE], digit_counts[CYCLE]);
	}
}


template <int SCAN_LANES_PER_CYCLE, int LOG_RAKING_THREADS_PER_LANE, int RAKING_THREADS_PER_LANE, int PARTIALS_PER_SEG>
__device__ __forceinline__ void PrefixScanOverLanes(
	int 	raking_segment[],
	int 	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int 	copy_section)
{
	// Upsweep rake
	int partial_reduction = SerialReduce<PARTIALS_PER_SEG>(raking_segment);

	// Warpscan reduction in digit warpscan_lane
	int warpscan_lane = threadIdx.x >> LOG_RAKING_THREADS_PER_LANE;
	int group_prefix = WarpScan<RAKING_THREADS_PER_LANE, true>(
		warpscan[warpscan_lane], 
		partial_reduction,
		copy_section);

	// Downsweep rake
	SerialScan<PARTIALS_PER_SEG>(raking_segment, group_prefix);
}


template <int SCAN_LANES_PER_CYCLE, int RAKING_THREADS_PER_LANE, int LOADS_PER_CYCLE, int SCAN_LANES_PER_LOAD>
__device__ __forceinline__ void RecoverDigitCounts(
	int warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int counts[LOADS_PER_CYCLE],
	int copy_section)
{
	int my_lane = threadIdx.x >> 2;
	int my_quad_byte = threadIdx.x & 3;
	
	#pragma unroll
	for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
		unsigned char *warpscan_count = (unsigned char *) &warpscan[my_lane + (SCAN_LANES_PER_LOAD * LOAD)][1 + copy_section][RAKING_THREADS_PER_LANE - 1];
		counts[LOAD] = warpscan_count[my_quad_byte];
	}
}

template<int RADIX_DIGITS>
__device__ __forceinline__ void CorrectLoadOverflow(
	int2 	load_digits,
	int 	&load_count)				
{
	if (WarpVoteAll(RADIX_DIGITS, load_count <= 1)) {
		// All keys have same digit 
		load_count = (threadIdx.x == load_digits.x) ? 256 : 0;
	}
}

template <int RADIX_DIGITS, int LOADS_PER_CYCLE>
__device__ __forceinline__ void CorrectCycleOverflow(
	int2 	cycle_digits[LOADS_PER_CYCLE],
	int 	cycle_counts[LOADS_PER_CYCLE])				
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (LOADS_PER_CYCLE > 0) CorrectLoadOverflow<RADIX_DIGITS>(cycle_digits[0], cycle_counts[0]);
	if (LOADS_PER_CYCLE > 1) CorrectLoadOverflow<RADIX_DIGITS>(cycle_digits[1], cycle_counts[1]);
}


template <int RADIX_DIGITS, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void CorrectTileOverflow(
	int2 	tile_digits[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int 	tile_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE])
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (CYCLES_PER_TILE > 0) CorrectCycleOverflow<RADIX_DIGITS, LOADS_PER_CYCLE>(tile_digits[0], tile_counts[0]);
	if (CYCLES_PER_TILE > 1) CorrectCycleOverflow<RADIX_DIGITS, LOADS_PER_CYCLE>(tile_digits[1], tile_counts[1]);
}


template <int RADIX_DIGITS>
__device__ __forceinline__ void CorrectLastLaneOverflow(int &count, const int &extra_elements) 
{
	if (WarpVoteAll(RADIX_DIGITS, count == 0) && (threadIdx.x == RADIX_DIGITS - 1)) {
		// We're 'f' and we overflowed b/c of invalid 'f' placemarkers; the number of valid items in this load is the count of valid f's 
		count = extra_elements & 255;
	}
}
		

template <int RADIX_DIGITS, int CYCLES_PER_TILE, int LOADS_PER_CYCLE, int LOADS_PER_TILE, bool UNGUARDED_IO>
__device__ __forceinline__ void CorrectForOverflows(
	int2 digits[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int counts[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	const int &extra_elements)				
{
	if (!UNGUARDED_IO) {
		// Correct any overflow in the partially-filled last lane
		int *linear_counts = (int *) counts;
		CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[LOADS_PER_TILE - 1], extra_elements);
	}

	CorrectTileOverflow<RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE>(digits, counts);
}


template <
	typename K,
	int BIT, 
	int RADIX_BITS,
	int RADIX_DIGITS,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int RAKING_THREADS,
	int SCAN_LANES_PER_CYCLE,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PADDED_PARTIALS_PER_LANE,
	int CYCLES_PER_TILE>
__device__ __forceinline__ void ScanCycle(
	int *base_partial,
	int	*raking_partial,
	int warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	typename VecType<K, 2>::Type keypairs[LOADS_PER_CYCLE],
	int2 digits[LOADS_PER_CYCLE],
	int2 flag_offsets[LOADS_PER_CYCLE],
	int2 ranks[LOADS_PER_CYCLE],
	int copy_section)
{
	// Reset smem
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_CYCLE; SCAN_LANE++) {
		base_partial[SCAN_LANE * PADDED_PARTIALS_PER_LANE] = 0;
	}
	
	// Decode digits for first cycle
	DecodeDigits<K, RADIX_BITS, BIT, LOADS_PER_CYCLE, SCAN_LANES_PER_LOAD, PADDED_PARTIALS_PER_LANE>(
		keypairs, digits, flag_offsets);
	
	// Encode counts into smem for first cycle
	PlacePartials<LOADS_PER_CYCLE>(
		(unsigned char *) base_partial,
		digits,
		flag_offsets); 
	
	__syncthreads();
	
	// Intra-group prefix scans for first cycle
	if (threadIdx.x < RAKING_THREADS) {
	
		PrefixScanOverLanes<SCAN_LANES_PER_CYCLE, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG>(		// first cycle is offset right by one
			raking_partial,
			warpscan, 
			copy_section);
	}
	
	__syncthreads();

	// Extract ranks
	ExtractRanks<LOADS_PER_CYCLE>(
		(unsigned char *) base_partial, 
		digits, 
		flag_offsets, 
		ranks); 	
}	
	

/******************************************************************************
 * SM1.3 Local Exchange Routines
 * 
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

template <typename T, bool UNGUARDED_IO, int CYCLES_PER_TILE, int LOADS_PER_CYCLE, typename PostprocessFunctor>
__device__ __forceinline__ void ScatterLoads(
	T *d_out, 
	typename VecType<T, 2>::Type pairs[LOADS_PER_CYCLE],
	int2 offsets[LOADS_PER_CYCLE],
	const int BASE4,
	const int &extra_elements,
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	#pragma unroll 
	for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
		postprocess(pairs[LOAD].x);
		postprocess(pairs[LOAD].y);

		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * (LOAD * 2 + 0)) < extra_elements)) 
			d_out[offsets[LOAD].x] = pairs[LOAD].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * (LOAD * 2 + 1)) < extra_elements)) 
			d_out[offsets[LOAD].y] = pairs[LOAD].y;
	}
}

template <typename T, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void PushPairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE])				
{
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
	
		#pragma unroll 
		for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
			swap[ranks[CYCLE][LOAD].x] = pairs[CYCLE][LOAD].x;
			swap[ranks[CYCLE][LOAD].y] = pairs[CYCLE][LOAD].y;
		}
	}
}
	
template <typename T, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void ExchangePairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE])				
{
	// Push in Pairs
	PushPairs<T, CYCLES_PER_TILE, LOADS_PER_CYCLE>(swap, pairs, ranks);
	
	__syncthreads();
	
	// Extract pairs
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
		
		#pragma unroll 
		for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			pairs[CYCLE][LOAD].x = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0))];
			pairs[CYCLE][LOAD].y = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1))];
		}
	}
}


template <
	typename K,
	typename V,	
	CacheModifier CACHE_MODIFIER,
	int RADIX_BITS,
	int RADIX_DIGITS, 
	int BIT, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm13(
	typename VecType<K, 2>::Type keypairs[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int *exchange,
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int digit_carry[RADIX_DIGITS], 
	const int &extra_elements)				
{
	int2 offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	
	// Swap keys according to ranks
	ExchangePairs<K, CYCLES_PER_TILE, LOADS_PER_CYCLE>((K*) exchange, keypairs, ranks);				
	
	// Calculate scatter offsets (re-decode digits from keys: it's less work than making a second exchange of digits)
	if (CYCLES_PER_TILE > 0) {
		const int CYCLE = 0;
		if (LOADS_PER_CYCLE > 0) {
			const int LOAD = 0;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
		if (LOADS_PER_CYCLE > 1) {
			const int LOAD = 1;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
	}
	if (CYCLES_PER_TILE > 1) {
		const int CYCLE = 1;
		if (LOADS_PER_CYCLE > 0) {
			const int LOAD = 0;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
		if (LOADS_PER_CYCLE > 1) {
			const int LOAD = 1;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<K, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
	}
	
	// Scatter keys
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
		const int BLOCK = CYCLE * LOADS_PER_CYCLE * 2;
		ScatterLoads<K, UNGUARDED_IO, CYCLES_PER_TILE, LOADS_PER_CYCLE, PostprocessFunctor>(d_out_keys, keypairs[CYCLE], offsets[CYCLE], B40C_RADIXSORT_THREADS * BLOCK, extra_elements);
	}

	if (!IsKeysOnly<V>()) {
	
		__syncthreads();

		// Read input data
		typename VecType<V, 2>::Type datapairs[CYCLES_PER_TILE][LOADS_PER_CYCLE];

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		if (CYCLES_PER_TILE > 0) ReadCycle<V, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<V> >::Read(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 0, extra_elements);
		if (CYCLES_PER_TILE > 1) ReadCycle<V, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<V> >::Read(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 1, extra_elements);
		
		// Swap data according to ranks
		ExchangePairs<V, CYCLES_PER_TILE, LOADS_PER_CYCLE>((V*) exchange, datapairs, ranks);
		
		// Scatter data
		#pragma unroll 
		for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
			const int BLOCK = CYCLE * LOADS_PER_CYCLE * 2;
			ScatterLoads<V, UNGUARDED_IO, CYCLES_PER_TILE, LOADS_PER_CYCLE, NopFunctor<V> >(d_out_values, datapairs[CYCLE], offsets[CYCLE], B40C_RADIXSORT_THREADS * BLOCK, extra_elements);
		}
	}
}


/******************************************************************************
 * SM1.0 Local Exchange Routines
 *
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

template <
	typename T, 
	int RADIX_DIGITS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor> 
__device__ __forceinline__ void ScatterCycle(
	T *swapmem,
	T *d_out, 
	int digit_scan[2][RADIX_DIGITS], 
	int digit_carry[RADIX_DIGITS], 
	const int &extra_elements,
	int base_digit,				
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	const int LOG_STORE_TXN_THREADS = B40C_LOG_MEM_BANKS(__CUDA_ARCH__);
	const int STORE_TXN_THREADS = 1 << LOG_STORE_TXN_THREADS;
	
	int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
	int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;
	
	int my_digit = base_digit + store_txn_digit;
	if (my_digit < RADIX_DIGITS) {
	
		int my_exclusive_scan = digit_scan[1][my_digit - 1];
		int my_inclusive_scan = digit_scan[1][my_digit];
		int my_digit_count = my_inclusive_scan - my_exclusive_scan;

		int my_carry = digit_carry[my_digit] + my_exclusive_scan;
		int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));
		
		while (my_aligned_offset < my_digit_count) {

			if ((my_aligned_offset >= 0) && (UNGUARDED_IO || (my_exclusive_scan + my_aligned_offset < extra_elements))) { 
			
				T datum = swapmem[my_exclusive_scan + my_aligned_offset];
				postprocess(datum);
				d_out[my_carry + my_aligned_offset] = datum;
			}
			my_aligned_offset += STORE_TXN_THREADS;
		}
	}
}

template <
	typename T,
	int RADIX_DIGITS, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterPairs(
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	T *exchange,
	T *d_out, 
	int digit_carry[RADIX_DIGITS], 
	int digit_scan[2][RADIX_DIGITS], 
	const int &extra_elements)				
{
	const int SCATTER_CYCLE_DIGITS = B40C_RADIXSORT_WARPS * (B40C_WARP_THREADS / B40C_MEM_BANKS(__CUDA_ARCH__));
	const int SCATTER_CYCLES = RADIX_DIGITS / SCATTER_CYCLE_DIGITS;

	// Push in pairs
	PushPairs<T, CYCLES_PER_TILE, LOADS_PER_CYCLE>(exchange, pairs, ranks);

	__syncthreads();

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, not an innermost loop"

	if (SCATTER_CYCLES > 0) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 0);
	if (SCATTER_CYCLES > 1) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 1);
	if (SCATTER_CYCLES > 2) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 2);
	if (SCATTER_CYCLES > 3) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 3);
	if (SCATTER_CYCLES > 4) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 4);
	if (SCATTER_CYCLES > 5) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 5);
	if (SCATTER_CYCLES > 6) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 6);
	if (SCATTER_CYCLES > 7) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, digit_carry, extra_elements, SCATTER_CYCLE_DIGITS * 7);
}


template <
	typename K,
	typename V,	
	CacheModifier CACHE_MODIFIER,
	int RADIX_DIGITS, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm10(
	typename VecType<K, 2>::Type keypairs[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int *exchange,
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int digit_carry[RADIX_DIGITS], 
	int digit_scan[2][RADIX_DIGITS], 
	const int &extra_elements)				
{
	// Swap and scatter keys
	SwapAndScatterPairs<K, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, ranks, (K*) exchange, d_out_keys, digit_carry, digit_scan, extra_elements);				
	
	if (!IsKeysOnly<V>()) {

		__syncthreads();
		
		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		// Read input data
		typename VecType<V, 2>::Type datapairs[CYCLES_PER_TILE][LOADS_PER_CYCLE];
		if (CYCLES_PER_TILE > 0) ReadCycle<V, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<V> >::Read(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 0, extra_elements);
		if (CYCLES_PER_TILE > 1) ReadCycle<V, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<V> >::Read(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 1, extra_elements);

		// Swap and scatter data
		SwapAndScatterPairs<V, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, NopFunctor<V> >(
			datapairs, ranks, (V*) exchange, d_out_values, digit_carry, digit_scan, extra_elements);				
	}
}


/******************************************************************************
 * Tile of RADIXSORT_TILE_ELEMENTS keys (and values)
 ******************************************************************************/

template <
	typename K,
	typename V,	
	CacheModifier CACHE_MODIFIER,
	int BIT, 
	bool UNGUARDED_IO,
	int RADIX_BITS,
	int RADIX_DIGITS,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int CYCLES_PER_TILE,
	int SCAN_LANES_PER_CYCLE,
	int RAKING_THREADS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PARTIALS_PER_ROW,
	int ROWS_PER_LANE,
	typename PreprocessFunctor,
	typename PostprocessFunctor>
__device__ __forceinline__ void ScanDigitTile(
	typename VecType<K, 2>::Type *d_in_keys, 
	typename VecType<V, 2>::Type *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int *exchange,								
	int	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int	digit_carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS],
	int	*base_partial,
	int	*raking_partial,
	const int &extra_elements)
{
	
	const int PADDED_PARTIALS_PER_LANE 		= ROWS_PER_LANE * (PARTIALS_PER_ROW + 1);	 
	const int LOADS_PER_TILE 				= CYCLES_PER_TILE * LOADS_PER_CYCLE;

	// N.B.: We use the following voodoo incantations to elide the compiler's miserable 
	// "declared but never referenced" warnings for these (which are actually used for 
	// template instantiation)	
	SuppressUnusedConstantWarning(PADDED_PARTIALS_PER_LANE);
	SuppressUnusedConstantWarning(LOADS_PER_TILE);
	
	typename VecType<K, 2>::Type 	keypairs[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int2 							digits[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int2 							flag_offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE];		// a byte offset
	int2 							ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE];

	
	//-------------------------------------------------------------------------
	// Read keys
	//-------------------------------------------------------------------------

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected control flow construct"
	
	// Read Keys
	if (CYCLES_PER_TILE > 0) ReadCycle<K, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, PreprocessFunctor>::Read(d_in_keys, keypairs[0], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 0, extra_elements);		 
	if (CYCLES_PER_TILE > 1) ReadCycle<K, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, PreprocessFunctor>::Read(d_in_keys, keypairs[1], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 1, extra_elements); 	
	
	//-------------------------------------------------------------------------
	// Lane-scanning Cycles
	//-------------------------------------------------------------------------

	if (CYCLES_PER_TILE > 0) {
		const int CYCLE = 0;
		ScanCycle<K, BIT, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, RAKING_THREADS, SCAN_LANES_PER_CYCLE, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PADDED_PARTIALS_PER_LANE, CYCLES_PER_TILE>(
			base_partial,
			raking_partial,
			warpscan,
			keypairs[CYCLE],
			digits[CYCLE],
			flag_offsets[CYCLE],
			ranks[CYCLE],
			CYCLES_PER_TILE - CYCLE - 1);		// lower cycles get copied right
	}
	if (CYCLES_PER_TILE > 1) {
		const int CYCLE = 1;
		ScanCycle<K, BIT, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, RAKING_THREADS, SCAN_LANES_PER_CYCLE, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PADDED_PARTIALS_PER_LANE, CYCLES_PER_TILE>(
			base_partial,
			raking_partial,
			warpscan,
			keypairs[CYCLE],
			digits[CYCLE],
			flag_offsets[CYCLE],
			ranks[CYCLE],
			CYCLES_PER_TILE - CYCLE - 1);		// lower cycles get copied right
	}
	
	
	//-------------------------------------------------------------------------
	// Digit-scanning 
	//-------------------------------------------------------------------------

	// Recover second-half digit-counts, scan across all digit-counts
	if (threadIdx.x < RADIX_DIGITS) {

		int counts[CYCLES_PER_TILE][LOADS_PER_CYCLE];

		// Recover digit-counts from warpscan padding

		#pragma unroll
		for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
			RecoverDigitCounts<SCAN_LANES_PER_CYCLE, RAKING_THREADS_PER_LANE, LOADS_PER_CYCLE, SCAN_LANES_PER_LOAD>(		// first cycle, offset by 1			
				warpscan, 
				counts[CYCLE],
				CYCLES_PER_TILE - CYCLE - 1);		// lower cycles get copied right
		}
		
		// Check for overflows
		CorrectForOverflows<RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, LOADS_PER_TILE, UNGUARDED_IO>(
			digits, counts, extra_elements);

		// Scan across my digit counts for each load 
		int exclusive_total = 0;
		int inclusive_total = 0;
		
		#pragma unroll
		for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
		
			#pragma unroll
			for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
				inclusive_total += counts[CYCLE][LOAD];
				counts[CYCLE][LOAD] = exclusive_total;
				exclusive_total = inclusive_total;
			}
		}

		// second half of digit_carry update
		int my_carry = digit_carry[threadIdx.x] + digit_scan[1][threadIdx.x];

		// Perform overflow-free SIMD Kogge-Stone across digits
		int digit_prefix = WarpScan<RADIX_DIGITS, false>(
				digit_scan, 
				inclusive_total,
				0);

		// first-half of digit_carry update 
		digit_carry[threadIdx.x] = my_carry - digit_prefix;
		
		#pragma unroll
		for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {

			#pragma unroll
			for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
				digit_counts[CYCLE][LOAD][threadIdx.x] = counts[CYCLE][LOAD] + digit_prefix;
			}
		}
	}
	
	__syncthreads();

	//-------------------------------------------------------------------------
	// Update Ranks
	//-------------------------------------------------------------------------

	UpdateRanks<RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE>(digits, ranks, digit_counts);
	
	
	//-------------------------------------------------------------------------
	// Scatter 
	//-------------------------------------------------------------------------

#if ((__CUDA_ARCH__ < 130) || FERMI_ECC)		

	SwapAndScatterSm10<K, V, CACHE_MODIFIER, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		digit_carry, 
		digit_scan,
		extra_elements);
	
#else 

	SwapAndScatterSm13<K, V, CACHE_MODIFIER, RADIX_BITS, RADIX_DIGITS, BIT, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		digit_carry, 
		extra_elements);
	
#endif

	__syncthreads();

}

template <
	typename K,
	typename V,	
	CacheModifier CACHE_MODIFIER,
	int BIT, 
	int RADIX_BITS,
	int RADIX_DIGITS,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int CYCLES_PER_TILE,
	int SCAN_LANES_PER_CYCLE,
	int RAKING_THREADS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PARTIALS_PER_ROW,
	int ROWS_PER_LANE,
	int TILE_ELEMENTS,
	typename PreprocessFunctor,
	typename PostprocessFunctor>
__device__ __forceinline__ void ScanScatterDigitPass(
	int *d_spine,
	K *d_in_keys, 
	V *d_in_values, 
	K *d_out_keys, 
	V *d_out_values, 
	int *exchange,								
	int	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int	digit_carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS],
	int	*base_partial,
	int	*raking_partial,		
	int block_offset,
	const int &out_of_bounds,
	const int &extra_elements)
{
	if (threadIdx.x < RADIX_DIGITS) {

		// Reset reused portion of digit_scan
		digit_scan[1][threadIdx.x] = 0;

		// Read digit_carry in parallel 
		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		int my_digit_carry;
		GlobalLoad<int, CACHE_MODIFIER>::Ld(my_digit_carry, d_spine, spine_digit_offset);
		digit_carry[threadIdx.x] = my_digit_carry;
	}

	// Scan in tiles of tile_elements
	while (block_offset < out_of_bounds) {
	
		ScanDigitTile<K, V, CACHE_MODIFIER, BIT, true, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE, PreprocessFunctor, PostprocessFunctor>(	
			reinterpret_cast<typename VecType<K, 2>::Type *>(&d_in_keys[block_offset]), 
			reinterpret_cast<typename VecType<V, 2>::Type *>(&d_in_values[block_offset]), 
			d_out_keys, 
			d_out_values, 
			exchange,
			warpscan,
			digit_carry,
			digit_scan,						 
			digit_counts,
			base_partial,
			raking_partial,
			extra_elements);	
	
		block_offset += TILE_ELEMENTS;
	}
	
	if (extra_elements) {
		
		// Clean up with guarded-io
		
		ScanDigitTile<K, V, CACHE_MODIFIER, BIT, false, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE, PreprocessFunctor, PostprocessFunctor>(	
			reinterpret_cast<typename VecType<K, 2>::Type *>(&d_in_keys[block_offset]), 
			reinterpret_cast<typename VecType<V, 2>::Type *>(&d_in_values[block_offset]), 
			d_out_keys, 
			d_out_values, 
			exchange,
			warpscan,
			digit_carry,
			digit_scan,						 
			digit_counts,
			base_partial,
			raking_partial,
			extra_elements);		
	}
}





template <
	typename K, 
	typename V, 
	int PASS, 
	int RADIX_BITS, 
	int BIT, 
	typename PreprocessFunctor, 
	typename PostprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void LsbScanScatterKernel(
	int *d_selectors,
	int* d_spine,
	K* d_keys0,
	K* d_keys1,
	V* d_values0,
	V* d_values1,
	CtaDecomposition work_decomposition)
{

	const int RADIX_DIGITS 				= 1 << RADIX_BITS;
	const int TILE_ELEMENTS				= B40C_RADIXSORT_TILE_ELEMENTS(__CUDA_ARCH__, K, V);
	
	const int LOG_SCAN_LANES_PER_LOAD	= (RADIX_BITS > 2) ? RADIX_BITS - 2 : 0;					// Always at one lane per load
	const int SCAN_LANES_PER_LOAD		= 1 << LOG_SCAN_LANES_PER_LOAD;								
	
	const int LOG_LOADS_PER_CYCLE		= B40C_RADIXSORT_LOG_LOADS_PER_CYCLE(__CUDA_ARCH__);			
	const int LOADS_PER_CYCLE			= 1 << LOG_LOADS_PER_CYCLE;
	
	const int LOG_CYCLES_PER_TILE		= B40C_RADIXSORT_LOG_CYCLES_PER_TILE(__CUDA_ARCH__, K, V);			
	const int CYCLES_PER_TILE			= 1 << LOG_CYCLES_PER_TILE;

	const int LOG_SCAN_LANES_PER_CYCLE	= LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD;
	const int SCAN_LANES_PER_CYCLE		= 1 << LOG_SCAN_LANES_PER_CYCLE;
	
	const int LOG_PARTIALS_PER_LANE 	= B40C_RADIXSORT_LOG_THREADS;
	
	const int LOG_PARTIALS_PER_CYCLE	= LOG_SCAN_LANES_PER_CYCLE + LOG_PARTIALS_PER_LANE;

	const int LOG_RAKING_THREADS 		= B40C_RADIXSORT_LOG_RAKING_THREADS(__CUDA_ARCH__);
	const int RAKING_THREADS			= 1 << LOG_RAKING_THREADS;

	const int LOG_RAKING_THREADS_PER_LANE 	= LOG_RAKING_THREADS - LOG_SCAN_LANES_PER_CYCLE;
	const int RAKING_THREADS_PER_LANE 		= 1 << LOG_RAKING_THREADS_PER_LANE;

	const int LOG_PARTIALS_PER_SEG 		= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE;
	const int PARTIALS_PER_SEG 			= 1 << LOG_PARTIALS_PER_SEG;

	const int LOG_PARTIALS_PER_ROW		= (LOG_PARTIALS_PER_SEG < B40C_LOG_MEM_BANKS(__CUDA_ARCH__)) ? B40C_LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of MEM_BANKS partials per row
	const int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	const int PADDED_PARTIALS_PER_ROW 	= PARTIALS_PER_ROW + 1;

	const int LOG_SEGS_PER_ROW 			= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	const int LOG_ROWS_PER_LOAD 			= LOG_PARTIALS_PER_CYCLE - LOG_PARTIALS_PER_ROW;

	const int LOG_ROWS_PER_LANE 		= LOG_PARTIALS_PER_LANE - LOG_PARTIALS_PER_ROW;
	const int ROWS_PER_LANE 			= 1 << LOG_ROWS_PER_LANE;

	const int LOG_ROWS_PER_CYCLE 		= LOG_SCAN_LANES_PER_CYCLE + LOG_ROWS_PER_LANE;
	const int ROWS_PER_CYCLE 			= 1 << LOG_ROWS_PER_CYCLE;
	
	const int SCAN_LANE_BYTES			= ROWS_PER_CYCLE * PADDED_PARTIALS_PER_ROW * sizeof(int);
	const int MAX_EXCHANGE_BYTES		= (sizeof(K) > sizeof(V)) ? 
													TILE_ELEMENTS * sizeof(K) : 
													TILE_ELEMENTS * sizeof(V);
	const int SCAN_LANE_INT4S         = (B40C_MAX(MAX_EXCHANGE_BYTES, SCAN_LANE_BYTES) + sizeof(int4) - 1) / sizeof(int4);


	// N.B.: We use the following voodoo incantations to elide the compiler's miserable 
	// "declared but never referenced" warnings for these (which are actually used for 
	// template instantiation)	
	SuppressUnusedConstantWarning(SCAN_LANES_PER_LOAD);
	SuppressUnusedConstantWarning(PARTIALS_PER_SEG);
	SuppressUnusedConstantWarning(LOG_ROWS_PER_LOAD);
	SuppressUnusedConstantWarning(ROWS_PER_LANE);

    // scan_lanes is a int4[] to avoid alignment issues when casting to (K *) and/or (V *)
	__shared__ int4		scan_lanes[SCAN_LANE_INT4S];
	__shared__ int 		warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE];		// One warpscan per fours-group
	__shared__ int 		digit_carry[RADIX_DIGITS];
	__shared__ int 		digit_scan[2][RADIX_DIGITS];						 
	__shared__ int 		digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS];
	__shared__ bool 	non_trivial_digit_pass;
	__shared__ int 		selector;
	
	_B40C_SCANSCATTER_REG_MISER_ int extra_elements;
	_B40C_SCANSCATTER_REG_MISER_ int out_of_bounds;


	// calculate our threadblock's range
	int block_elements, block_offset;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		block_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		block_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * TILE_ELEMENTS);
		block_elements = work_decomposition.normal_block_elements;
	}
	extra_elements = 0;
	if (blockIdx.x == gridDim.x - 1) {
		extra_elements = work_decomposition.extra_elements_last_block;
		if (extra_elements) {
			block_elements -= TILE_ELEMENTS;
		}
	}
	out_of_bounds = block_offset + block_elements;	
	
	
	// location for placing 2-element partial reductions in the first lane of a cycle	
	int row = threadIdx.x >> LOG_PARTIALS_PER_ROW; 
	int col = threadIdx.x & (PARTIALS_PER_ROW - 1); 
	int *base_partial = reinterpret_cast<int *>(scan_lanes) + (row * PADDED_PARTIALS_PER_ROW) + col; 								
	
	// location for raking across all loads within a cycle
	int *raking_partial = 0;										

	if (threadIdx.x < RAKING_THREADS) {

		// initalize lane warpscans
		if (threadIdx.x < RAKING_THREADS_PER_LANE) {
			
			#pragma unroll
			for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_CYCLE; SCAN_LANE++) {
				warpscan[SCAN_LANE][0][threadIdx.x] = 0;
			}
		}

		// initialize digit warpscans
		if (threadIdx.x < RADIX_DIGITS) {

			// Initialize digit_scan
			digit_scan[0][threadIdx.x] = 0;

			// Determine where to read our input
			selector = (PASS == 0) ? 0 : d_selectors[PASS & 0x1];

			if (PreprocessFunctor::MustApply() || PostprocessFunctor::MustApply()) {
				non_trivial_digit_pass = true;

			} else {

				// Determine whether or not we have work to do and setup the next round 
				// accordingly.  We can do this by looking at the first-block's 
				// histograms and counting the number of digits with counts that are 
				// non-zero and not-the-problem-size.

				int first_block_carry = d_spine[FastMul(gridDim.x, threadIdx.x)];
				int predicate = ((first_block_carry > 0) && (first_block_carry < work_decomposition.num_elements));
				non_trivial_digit_pass = (TallyWarpVote(RADIX_DIGITS, predicate, reinterpret_cast<int *>(scan_lanes)) > 0);
			}

			// Let the next round know which set of buffers to use
			if (blockIdx.x == 0) {
				d_selectors[(PASS + 1) & 0x1] = selector ^ non_trivial_digit_pass;
			}
		}

		// initialize raking segment
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		raking_partial = reinterpret_cast<int *>(scan_lanes) + (row * PADDED_PARTIALS_PER_ROW) + col; 
	}

	// Sync to acquire non_trivial_digit_pass and selector
	__syncthreads();
	
	// Short-circuit this entire cycle
	if (!non_trivial_digit_pass) return; 

	if (!selector) {
	
		// d_keys0 -> d_keys1 
		ScanScatterDigitPass<K, V, NONE, BIT, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor>(	
			d_spine,
			d_keys0, 
			d_values0, 
			d_keys1, 
			d_values1, 
			(int *) scan_lanes,
			warpscan,
			digit_carry,
			digit_scan,						 
			digit_counts,
			base_partial,
			raking_partial,
			block_offset,
			out_of_bounds,
			extra_elements);		
	
	} else {
		
		// d_keys1 -> d_keys0
		ScanScatterDigitPass<K, V, NONE, BIT, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor>(	
			d_spine,
			d_keys1, 
			d_values1, 
			d_keys0, 
			d_values0, 
			(int *) scan_lanes,
			warpscan,
			digit_carry,
			digit_scan,						 
			digit_counts,
			base_partial,
			raking_partial,
			block_offset,
			out_of_bounds,
			extra_elements);		
	}
}


} // namespace b40c

