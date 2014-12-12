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
// Bottom-level digit scanning/scattering kernel
 ******************************************************************************/

#pragma once

#include "radixsort_reduction_kernel.cu"
#include "radixsort_spine_kernel.cu"
#include "radixsort_scanscatter_kernel.cu"

namespace b40c {


/******************************************************************************
 * Single-grid (SG) kernel for LSB radix sorting
 ******************************************************************************/
	

// Target threadblock occupancy for bulk scan/scatter kernel
#define B40C_SM20_SG_OCCUPANCY()								(6)			// 8 threadblocks on GF100
#define B40C_SM12_SG_OCCUPANCY()								(1)			// 8 threadblocks on GT200
#define B40C_SM10_SG_OCCUPANCY()								(1)			// 8 threadblocks on G80
#define B40C_RADIXSORT_SG_OCCUPANCY(version)					((version >= 200) ? B40C_SM20_SG_OCCUPANCY() : 	\
																 (version >= 120) ? B40C_SM12_SG_OCCUPANCY() : 	\
																					B40C_SM10_SG_OCCUPANCY())		

// Number of 256-element loads to rake per raking cycle
#define B40C_SM20_SG_LOG_LOADS_PER_CYCLE(K, V)					(1)			// 2 loads on GF100 
#define B40C_SM12_SG_LOG_LOADS_PER_CYCLE(K, V)					(1)			// 2 loads on GT200
#define B40C_SM10_SG_LOG_LOADS_PER_CYCLE(K, V)					(1)			// 2 loads on G80
#define B40C_RADIXSORT_SG_LOG_LOADS_PER_CYCLE(version, K, V)	((version >= 200) ? B40C_SM20_SG_LOG_LOADS_PER_CYCLE(K, V) : 	\
																 (version >= 120) ? B40C_SM12_SG_LOG_LOADS_PER_CYCLE(K, V) : 	\
																					B40C_SM10_SG_LOG_LOADS_PER_CYCLE(K, V))		

// Number of raking cycles per tile
#define B40C_SM20_SG_LOG_CYCLES_PER_TILE(K, V)					(0)			// 1 cycle on GF100
#define B40C_SM12_SG_LOG_CYCLES_PER_TILE(K, V)					(0)			// 1 cycle on GT200
#define B40C_SM10_SG_LOG_CYCLES_PER_TILE(K, V)					(0)			// 1 cycle on G80
#define B40C_RADIXSORT_SG_LOG_CYCLES_PER_TILE(version, K, V)	((version >= 200) ? B40C_SM20_SG_LOG_CYCLES_PER_TILE(K, V) : 	\
																 (version >= 120) ? B40C_SM12_SG_LOG_CYCLES_PER_TILE(K, V) : 	\
																					B40C_SM10_SG_LOG_CYCLES_PER_TILE(K, V))		

// Number of raking threads per raking cycle
#define B40C_SM20_SG_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS + 2)		// 2 raking warps on GF100
#define B40C_SM12_SG_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS + 2)		// 1 raking warp on GT200
#define B40C_SM10_SG_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS + 2)		// 4 raking warps on G80
#define B40C_RADIXSORT_SG_LOG_RAKING_THREADS(version)			((version >= 200) ? B40C_SM20_SG_LOG_RAKING_THREADS() : 	\
																 (version >= 120) ? B40C_SM12_SG_LOG_RAKING_THREADS() : 	\
																					B40C_SM10_SG_LOG_RAKING_THREADS())		

// Number of elements per tile
#define B40C_RADIXSORT_SG_LOG_TILE_ELEMENTS(version, K, V)	(B40C_RADIXSORT_SG_LOG_LOADS_PER_CYCLE(version, K, V) + B40C_RADIXSORT_SG_LOG_CYCLES_PER_TILE(version, K, V) + B40C_RADIXSORT_LOG_THREADS + 1)
#define B40C_RADIXSORT_SG_TILE_ELEMENTS(version, K, V)		(1 << B40C_RADIXSORT_SG_LOG_TILE_ELEMENTS(version, K, V))



__device__ __forceinline__ int LoadCG(int* d_ptr) 
{
	int retval;
	GlobalLoad<int, CG>::Ld(retval, d_ptr, 0);
	return retval;
}


/**
 * Implements a global, lock-free software barrier between CTAs
 */
__device__ __forceinline__ void GlobalBarrier(int* d_sync) 
{
	// Threadfence and syncthreads to make sure global writes are visible before 
	// thread-0 reports in with its sync counter
	__threadfence();
	__syncthreads();
	
	if (blockIdx.x == 0) {

		// Report in ourselves
		if (threadIdx.x == 0) {
			d_sync[blockIdx.x] = 1; 
		}

		__syncthreads();
		
		// Wait for everyone else to report in
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += B40C_RADIXSORT_THREADS) {
			while (LoadCG(d_sync + peer_block) == 0) {
				__threadfence_block();
			}
		}

		__syncthreads();
		
		// Let everyone know it's safe to read their prefix sums
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += B40C_RADIXSORT_THREADS) {
			d_sync[peer_block] = 0;
		}

	} else {
		
		if (threadIdx.x == 0) {
			// Report in 
			d_sync[blockIdx.x] = 1; 

			// Wait for acknowledgement
			while (LoadCG(d_sync + blockIdx.x) == 1) {
				__threadfence_block();
			}
		}
		
		__syncthreads();
	}
}


template <
	typename K, 
	typename V,
	int BIT, 
	int RADIX_BITS,
	int RADIX_DIGITS,
	int TILE_ELEMENTS,
	typename PreprocessFunctor, 
	typename PostprocessFunctor,
	int REDUCTION_LANES,
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE,
	int SPINE_PARTIALS_PER_SEG,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int CYCLES_PER_TILE,
	int SCAN_LANES_PER_CYCLE,
	int RAKING_THREADS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PARTIALS_PER_ROW,
	int ROWS_PER_LANE>
__device__ __forceinline__ void DistributionSortingPass(
	int* d_sync,
	int* d_spine,
	K* d_in_keys,
	K* d_out_keys,
	V* d_in_values,
	V* d_out_values,
	int block_offset,
	int block_elements,
	const int &out_of_bounds,
	const int &extra_elements,
	int spine_elements,
	int *base_partial,
	int *raking_partial,
	int *spine_raking_partial,
	int *encoded_reduction_col,
	int *smem_pool,
	int	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int	digit_carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS],
	int spine_scan[2][B40C_WARP_THREADS])
{
	//-------------------------------------------------------------------------
	// Reduction
	//-------------------------------------------------------------------------

	ReductionPass<K, CG, BIT, RADIX_BITS, RADIX_DIGITS, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, PreprocessFunctor, false>(
		d_in_keys,
		d_spine,
		block_offset,
		encoded_reduction_col,
		smem_pool,
		out_of_bounds + extra_elements);

	
	//-------------------------------------------------------------------------
	// Global Barrier + Scan spine
	//-------------------------------------------------------------------------

	// Threadfence and syncthreads to make sure global writes are visible before 
	// thread-0 reports in with its sync counter
	__threadfence();
	__syncthreads();
	
	if (blockIdx.x == 0) {

		// Report in ourselves
		if (threadIdx.x == 0) {
			d_sync[blockIdx.x] = 1; 
		}

		// Wait for everyone else to report in
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += B40C_RADIXSORT_THREADS) {
			while (LoadCG(d_sync + peer_block) == 0) {
				__threadfence_block();
			}
		}

		__syncthreads();

		// Scan the spine in blocks of tile_elements
		int spine_carry = 0;
		int spine_offset = 0;
		while (spine_offset < spine_elements) {
			
			SrtsScanTile<CG, SPINE_PARTIALS_PER_SEG>(	
				base_partial, 
				spine_raking_partial, 
				spine_scan,
				reinterpret_cast<int4 *>(&d_spine[spine_offset]), 
				reinterpret_cast<int4 *>(&d_spine[spine_offset]), 
				spine_carry);
	
			spine_offset += B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		}
		
		// Threadfence and syncthreads to make sure global writes are visible before 
		// everyone reports back with their sync counters
		__threadfence();
		__syncthreads();

		// Let everyone know it's safe to read their prefix sums
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += B40C_RADIXSORT_THREADS) {
			d_sync[peer_block] = 0;
		}

	} else {
		
		if (threadIdx.x == 0) {
			// Report in 
			d_sync[blockIdx.x] = 1; 

			// Wait for acknowledgement
			while (LoadCG(d_sync + blockIdx.x) == 1) {
				__threadfence_block();
			}
		}
		
		__syncthreads();
	}

	
	//-------------------------------------------------------------------------
	// Scan/Scatter
	//-------------------------------------------------------------------------

	ScanScatterDigitPass<K, V, CG, BIT, RADIX_BITS, RADIX_DIGITS, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor>(	
		d_spine,
		d_in_keys, 
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		smem_pool,
		warpscan,
		digit_carry,
		digit_scan,						 
		digit_counts,
		base_partial,
		raking_partial,
		block_offset,
		out_of_bounds,
		extra_elements);
	
	//-------------------------------------------------------------------------
	// Global barrier
	//-------------------------------------------------------------------------

	GlobalBarrier(d_sync);	
}



template <
	int PASS,
	typename K, 
	typename V,
	int RADIX_BITS,
	int RADIX_DIGITS,
	int TILE_ELEMENTS,
	typename PreprocessFunctor, 
	typename PostprocessFunctor,
	int REDUCTION_LANES,
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE,
	int SPINE_PARTIALS_PER_SEG,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int CYCLES_PER_TILE,
	int SCAN_LANES_PER_CYCLE,
	int RAKING_THREADS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PARTIALS_PER_ROW,
	int ROWS_PER_LANE>
__device__ __forceinline__ void DistributionSortingPass(
	int* d_sync,
	int* d_spine,
	K* d_keys0,
	K* d_keys1,
	V* d_values0,
	V* d_values1,
	int block_offset,
	int block_elements,
	const int &out_of_bounds,
	const int &extra_elements,
	int spine_elements,
	int *base_partial,
	int *raking_partial,
	int *spine_raking_partial,
	int *encoded_reduction_col,
	int *smem_pool,
	int	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int	digit_carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS],
	int spine_scan[2][B40C_WARP_THREADS])
{
	const int BIT = PASS * RADIX_BITS;

	SuppressUnusedConstantWarning(BIT);
	
	if (PASS & 0x1) {
		// Odd pass (flip keys0/keys1)
		DistributionSortingPass<K, V, BIT, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys1, d_keys0, d_values1, d_values0, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	} else {
		// Even pass
		DistributionSortingPass<K, V, BIT, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan); 
	}
}


template <
	int PASSES,
	int PASS,
	typename K, 
	typename V,
	int RADIX_BITS,
	int RADIX_DIGITS,
	int TILE_ELEMENTS,
	typename PreprocessFunctor, 
	typename PostprocessFunctor,
	int REDUCTION_LANES,
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE,
	int SPINE_PARTIALS_PER_SEG,
	int SCAN_LANES_PER_LOAD,
	int LOADS_PER_CYCLE,
	int CYCLES_PER_TILE,
	int SCAN_LANES_PER_CYCLE,
	int RAKING_THREADS,
	int LOG_RAKING_THREADS_PER_LANE,
	int RAKING_THREADS_PER_LANE,
	int PARTIALS_PER_SEG,
	int PARTIALS_PER_ROW,
	int ROWS_PER_LANE>
__device__ __forceinline__ void DistributionSortingPass(
	int* d_sync,
	int* d_spine,
	K* d_keys0,
	K* d_keys1,
	V* d_values0,
	V* d_values1,
	int block_offset,
	int block_elements,
	const int &out_of_bounds,
	const int &extra_elements,
	int spine_elements,
	int *base_partial,
	int *raking_partial,
	int *spine_raking_partial,
	int *encoded_reduction_col,
	int *smem_pool,
	int	warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE],
	int	digit_carry[RADIX_DIGITS],
	int	digit_scan[2][RADIX_DIGITS],						 
	int	digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS],
	int spine_scan[2][B40C_WARP_THREADS])
{
	if (PASSES == 1) { 

		// Only one pass: use both key pre- and post- processors in the same pass 
		DistributionSortingPass<PASS, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan); 

	} else if (PASS == 0) {
		
		// First pass: use key pre-processor in this pass
		DistributionSortingPass<PASS, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, NopFunctor<K>, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan); 

	} else if (PASS == PASSES - 1) {
		
		// Last pass: use key post-processor in this pass
		DistributionSortingPass<PASS, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, NopFunctor<K>, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan); 

	} else {
		
		// Middle pass: use nop-functors for keys this pass
		DistributionSortingPass<PASS, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, NopFunctor<K>, NopFunctor<K>, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan); 
	}
}


/**
 * Single-grid sorting kernel.  Performs up to 8 passes.
 */
template <
	typename K, 
	typename V, 
	int RADIX_BITS, 
	int PASSES,
	int STARTING_PASS,
	typename PreprocessFunctor, 
	typename PostprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_SG_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void LsbSingleGridSortingKernel(
	int* d_sync,
	int* d_spine,
	K* d_keys0,
	K* d_keys1,
	V* d_values0,
	V* d_values1,
	CtaDecomposition work_decomposition,
	int spine_elements)
{
	const int RADIX_DIGITS 					= 1 << RADIX_BITS;
	const int TILE_ELEMENTS					= B40C_RADIXSORT_SG_TILE_ELEMENTS(__CUDA_ARCH__, K, V);
	
	const int LOG_SCAN_LANES_PER_LOAD		= (RADIX_BITS > 2) ? RADIX_BITS - 2 : 0;					// Always at one lane per load
	const int SCAN_LANES_PER_LOAD			= 1 << LOG_SCAN_LANES_PER_LOAD;								
	
	const int LOG_LOADS_PER_CYCLE			= B40C_RADIXSORT_SG_LOG_LOADS_PER_CYCLE(__CUDA_ARCH__, K, V);			
	const int LOADS_PER_CYCLE				= 1 << LOG_LOADS_PER_CYCLE;
	
	const int LOG_CYCLES_PER_TILE			= B40C_RADIXSORT_SG_LOG_CYCLES_PER_TILE(__CUDA_ARCH__, K, V);			
	const int CYCLES_PER_TILE				= 1 << LOG_CYCLES_PER_TILE;

	const int LOG_SCAN_LANES_PER_CYCLE		= LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD;
	const int SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE;
	
	const int LOG_PARTIALS_PER_LANE 		= B40C_RADIXSORT_LOG_THREADS;
	
	const int LOG_PARTIALS_PER_CYCLE		= LOG_SCAN_LANES_PER_CYCLE + LOG_PARTIALS_PER_LANE;

	const int LOG_RAKING_THREADS 			= B40C_RADIXSORT_SG_LOG_RAKING_THREADS(__CUDA_ARCH__);
	const int RAKING_THREADS				= 1 << LOG_RAKING_THREADS;

	const int LOG_RAKING_THREADS_PER_LANE 	= LOG_RAKING_THREADS - LOG_SCAN_LANES_PER_CYCLE;
	const int RAKING_THREADS_PER_LANE 		= 1 << LOG_RAKING_THREADS_PER_LANE;

	const int LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE;
	const int PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG;

	const int LOG_PARTIALS_PER_ROW			= (LOG_PARTIALS_PER_SEG < B40C_LOG_MEM_BANKS(__CUDA_ARCH__)) ? B40C_LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of MEM_BANKS partials per row
	const int PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW;
	const int PADDED_PARTIALS_PER_ROW 		= PARTIALS_PER_ROW + 1;

	const int LOG_SEGS_PER_ROW 				= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const int SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW;

	const int LOG_ROWS_PER_LOAD 			= LOG_PARTIALS_PER_CYCLE - LOG_PARTIALS_PER_ROW;

	const int LOG_ROWS_PER_LANE 			= LOG_PARTIALS_PER_LANE - LOG_PARTIALS_PER_ROW;
	const int ROWS_PER_LANE 				= 1 << LOG_ROWS_PER_LANE;

	const int LOG_ROWS_PER_CYCLE 			= LOG_SCAN_LANES_PER_CYCLE + LOG_ROWS_PER_LANE;
	const int ROWS_PER_CYCLE 				= 1 << LOG_ROWS_PER_CYCLE;
	
	const int REDUCTION_LANES 					= SCAN_LANES_PER_LOAD;
	const int LOG_REDUCTION_PARTIALS_PER_LANE	= B40C_RADIXSORT_LOG_THREADS;
	const int REDUCTION_PARTIALS_PER_LANE 		= 1 << LOG_PARTIALS_PER_LANE;
	
	const int PADDED_REDUCTION_PARTIALS 	= B40C_WARP_THREADS + 1;
	
	const int SCAN_LANE_BYTES				= ROWS_PER_CYCLE * PADDED_PARTIALS_PER_ROW * sizeof(int);
	const int REDUCTION_LANE_BYTES			= REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE * sizeof(int);
	const int REDUCTION_RAKING_BYTES		= RADIX_DIGITS * PADDED_REDUCTION_PARTIALS;
	const int MAX_EXCHANGE_BYTES			= B40C_MAX(TILE_ELEMENTS * sizeof(K), TILE_ELEMENTS * sizeof(V));

	const int SHARED_BYTES 					= B40C_MAX(REDUCTION_RAKING_BYTES, B40C_MAX(SCAN_LANE_BYTES, B40C_MAX(REDUCTION_LANE_BYTES, MAX_EXCHANGE_BYTES)));
	const int SHARED_INT4S					= (SHARED_BYTES + sizeof(int4) - 1) / sizeof(int4);

	const int LOG_SPINE_RAKING_THREADS 		= B40C_LOG_WARP_THREADS;
	const int SPINE_RAKING_THREADS			= 1 << LOG_SPINE_RAKING_THREADS;

	const int LOG_SPINE_PARTIALS			= B40C_RADIXSORT_LOG_THREADS;				
	const int SPINE_PARTIALS			 	= 1 << LOG_SPINE_PARTIALS;
	
	const int LOG_SPINE_PARTIALS_PER_SEG 	= LOG_SPINE_PARTIALS - LOG_SPINE_RAKING_THREADS;	
	const int SPINE_PARTIALS_PER_SEG 		= 1 << LOG_SPINE_PARTIALS_PER_SEG;

	const int LOG_SPINE_SEGS_PER_ROW 		= LOG_PARTIALS_PER_ROW - LOG_SPINE_PARTIALS_PER_SEG;	
	const int SPINE_SEGS_PER_ROW			= 1 << LOG_SPINE_SEGS_PER_ROW;

	
	// N.B.: We use the following voodoo incantations to elide the compiler's miserable 
	// "declared but never referenced" warnings for these (which are actually used for 
	// template instantiation)	
	SuppressUnusedConstantWarning(SCAN_LANES_PER_LOAD);
	SuppressUnusedConstantWarning(PARTIALS_PER_SEG);
	SuppressUnusedConstantWarning(LOG_ROWS_PER_LOAD);
	SuppressUnusedConstantWarning(ROWS_PER_LANE);
	SuppressUnusedConstantWarning(LOG_REDUCTION_PARTIALS_PER_LANE);
	SuppressUnusedConstantWarning(SPINE_RAKING_THREADS);
	SuppressUnusedConstantWarning(SPINE_PARTIALS);
	SuppressUnusedConstantWarning(SPINE_PARTIALS_PER_SEG);

	__shared__ int4		aligned_smem_pool[SHARED_INT4S];								// aligned_smem_pool is a int4[] to avoid alignment issues when casting to (K *) and/or (V *)
	__shared__ int 		warpscan[SCAN_LANES_PER_CYCLE][3][RAKING_THREADS_PER_LANE];		// One warpscan per fours-group
	__shared__ int 		digit_carry[RADIX_DIGITS];
	__shared__ int 		digit_scan[2][RADIX_DIGITS];						 
	__shared__ int 		digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS];
	__shared__ int 		spine_scan[2][B40C_WARP_THREADS];
	                  
	__shared__ int extra_elements;
	__shared__ int out_of_bounds;

	int* smem_pool = reinterpret_cast<int*>(aligned_smem_pool);
	
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
	
	// Column for encoding reduction counts 
	int* encoded_reduction_col = smem_pool + threadIdx.x;	// first element of column

	// Location for placing 2-element partial reductions in the first lane of a cycle	
	int row = threadIdx.x >> LOG_PARTIALS_PER_ROW; 
	int col = threadIdx.x & (PARTIALS_PER_ROW - 1); 
	int *base_partial = smem_pool + (row * PADDED_PARTIALS_PER_ROW) + col; 								
	
	int *spine_raking_partial = 0;
	int *raking_partial = 0;										
	if (threadIdx.x < RAKING_THREADS) {

		if (threadIdx.x < B40C_WARP_THREADS) {
			// Location for spine raking
			row = threadIdx.x >> LOG_SPINE_SEGS_PER_ROW;
			col = (threadIdx.x & (SPINE_SEGS_PER_ROW - 1)) << LOG_SPINE_PARTIALS_PER_SEG;
			spine_raking_partial = smem_pool + (row * PADDED_PARTIALS_PER_ROW) + col; 

			// Initialize warpscan for spine_scan
			spine_scan[0][threadIdx.x] = 0;
		}

		// Location for scan/scatter-raking across all loads within a cycle
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		raking_partial = smem_pool + (row * PADDED_PARTIALS_PER_ROW) + col; 

		// Initalize lane warpscans
		if (threadIdx.x < RAKING_THREADS_PER_LANE) {
			
			#pragma unroll
			for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_CYCLE; SCAN_LANE++) {
				warpscan[SCAN_LANE][0][threadIdx.x] = 0;
			}
		}

		// Initialize digit_scan
		if (threadIdx.x < RADIX_DIGITS) {
			digit_scan[0][threadIdx.x] = 0;
		}
	}

	// Up to 8 sorting passes
	
	if (PASSES > 0) {
		DistributionSortingPass<PASSES, STARTING_PASS + 0, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 1) {
		DistributionSortingPass<PASSES, STARTING_PASS + 1, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 2) {
		DistributionSortingPass<PASSES, STARTING_PASS + 2, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 3) {
		DistributionSortingPass<PASSES, STARTING_PASS + 3, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 4) {
		DistributionSortingPass<PASSES, STARTING_PASS + 4, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 5) {
		DistributionSortingPass<PASSES, STARTING_PASS + 5, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 6) {
		DistributionSortingPass<PASSES, STARTING_PASS + 6, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
	if (PASSES > 7) {
		DistributionSortingPass<PASSES, STARTING_PASS + 7, K, V, RADIX_BITS, RADIX_DIGITS, TILE_ELEMENTS, PreprocessFunctor, PostprocessFunctor, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, SPINE_PARTIALS_PER_SEG, SCAN_LANES_PER_LOAD, LOADS_PER_CYCLE, CYCLES_PER_TILE, SCAN_LANES_PER_CYCLE, RAKING_THREADS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PARTIALS_PER_ROW, ROWS_PER_LANE>(  
			d_sync, d_spine, d_keys0, d_keys1, d_values0, d_values1, block_offset, block_elements, out_of_bounds, extra_elements, spine_elements, base_partial, raking_partial, spine_raking_partial, encoded_reduction_col, smem_pool, warpscan, digit_carry, digit_scan, digit_counts, spine_scan);
	}
}


} // namespace b40c

