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
 * Configuration management for B40C radix sorting kernels  
 ******************************************************************************/

#pragma once

#include <voxelpipe/b40c/KernelCommon/b40c_kernel_utils.cu>
#include <voxelpipe/b40c/KernelCommon/b40c_vector_types.cu>
#include <voxelpipe/b40c/LsbRadixSort/kernel/radixsort_key_conversion.cu>

namespace b40c {


/******************************************************************************
 * Radix sorting configuration  
 ******************************************************************************/

#define B40C_RADIXSORT_LOG_THREADS							(7)			// 128 threads								
#define B40C_RADIXSORT_THREADS								(1 << B40C_RADIXSORT_LOG_THREADS)	

// Target threadblock occupancy for counting/reduction kernel
#define B40C_SM20_REDUCE_CTA_OCCUPANCY()					(8)			// 8 threadblocks on GF100
#define B40C_SM12_REDUCE_CTA_OCCUPANCY()					(5)			// 5 threadblocks on GT200
#define B40C_SM10_REDUCE_CTA_OCCUPANCY()					(3)			// 3 threadblocks on G80
#define B40C_RADIXSORT_REDUCE_CTA_OCCUPANCY(version)		((version >= 200) ? B40C_SM20_REDUCE_CTA_OCCUPANCY() : 	\
			        										 (version >= 120) ? B40C_SM12_REDUCE_CTA_OCCUPANCY() : 	\
					        													B40C_SM10_REDUCE_CTA_OCCUPANCY())		
													                    
// Target threadblock occupancy for bulk scan/scatter kernel
#define B40C_SM20_SCAN_SCATTER_CTA_OCCUPANCY()				(7)			// 7 threadblocks on GF100
#define B40C_SM12_SCAN_SCATTER_CTA_OCCUPANCY()				(5)			// 5 threadblocks on GT200
#define B40C_SM10_SCAN_SCATTER_CTA_OCCUPANCY()				(2)			// 2 threadblocks on G80
#define B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(version)	((version >= 200) ? B40C_SM20_SCAN_SCATTER_CTA_OCCUPANCY() : 	\
			    											 (version >= 120) ? B40C_SM12_SCAN_SCATTER_CTA_OCCUPANCY() : 	\
				    															B40C_SM10_SCAN_SCATTER_CTA_OCCUPANCY())		

// Number of 256-element sets to rake per raking cycle
#define B40C_SM20_LOG_LOADS_PER_CYCLE()						(1)			// 2 sets on GF100
#define B40C_SM12_LOG_LOADS_PER_CYCLE()						(0)			// 1 set on GT200
#define B40C_SM10_LOG_LOADS_PER_CYCLE()						(1)			// 2 sets on G80
#define B40C_RADIXSORT_LOG_LOADS_PER_CYCLE(version)			((version >= 200) ? B40C_SM20_LOG_LOADS_PER_CYCLE() : 	\
															 (version >= 120) ? B40C_SM12_LOG_LOADS_PER_CYCLE() : 	\
				    														B40C_SM10_LOG_LOADS_PER_CYCLE())		

// Number of raking cycles per tile
#define B40C_SM20_LOG_CYCLES_PER_TILE(K, V)					(B40C_MAX(sizeof(K), sizeof(V)) > 4 ? 0 : 1)	// 2 cycles on GF100 (only one for large keys/values)
#define B40C_SM12_LOG_CYCLES_PER_TILE(K, V)					(B40C_MAX(sizeof(K), sizeof(V)) > 4 ? 0 : 1)	// 2 cycles on GT200 (only one for large keys/values)
#define B40C_SM10_LOG_CYCLES_PER_TILE(K, V)					(0)												// 1 cycle on G80
#define B40C_RADIXSORT_LOG_CYCLES_PER_TILE(version, K, V)	((version >= 200) ? B40C_SM20_LOG_CYCLES_PER_TILE(K, V) : 	\
				    										 (version >= 120) ? B40C_SM12_LOG_CYCLES_PER_TILE(K, V) : 	\
					    														B40C_SM10_LOG_CYCLES_PER_TILE(K, V))		

// Number of raking threads per raking cycle
#define B40C_SM20_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS + 1)		// 2 raking warps on GF100
#define B40C_SM12_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS)			// 1 raking warp on GT200
#define B40C_SM10_LOG_RAKING_THREADS()						(B40C_LOG_WARP_THREADS + 2)		// 4 raking warps on G80
#define B40C_RADIXSORT_LOG_RAKING_THREADS(version)			((version >= 200) ? B40C_SM20_LOG_RAKING_THREADS() : 	\
				    										 (version >= 120) ? B40C_SM12_LOG_RAKING_THREADS() : 	\
					    														B40C_SM10_LOG_RAKING_THREADS())		

// Number of elements per tile
#define B40C_RADIXSORT_LOG_TILE_ELEMENTS(version, K, V)		(B40C_RADIXSORT_LOG_LOADS_PER_CYCLE(version) + B40C_RADIXSORT_LOG_CYCLES_PER_TILE(version, K, V) + B40C_RADIXSORT_LOG_THREADS + 1)
#define B40C_RADIXSORT_TILE_ELEMENTS(version, K, V)			(1 << B40C_RADIXSORT_LOG_TILE_ELEMENTS(version, K, V))

// Number of warps per CTA
#define B40C_RADIXSORT_LOG_WARPS							(B40C_RADIXSORT_LOG_THREADS - B40C_LOG_WARP_THREADS)
#define B40C_RADIXSORT_WARPS								(1 << B40C_RADIXSORT_LOG_WARPS)

// Number of threads for spine-scanning kernel
#define B40C_RADIXSORT_LOG_SPINE_THREADS					(7)		// 128 threads
#define B40C_RADIXSORT_SPINE_THREADS						(1 << B40C_RADIXSORT_LOG_SPINE_THREADS)	

// Number of elements per spine-scanning tile
#define B40C_RADIXSORT_LOG_SPINE_TILE_ELEMENTS  			(B40C_RADIXSORT_LOG_SPINE_THREADS + 2)			// 512 elements
#define B40C_RADIXSORT_SPINE_TILE_ELEMENTS		    		(1 << B40C_RADIXSORT_LOG_SPINE_TILE_ELEMENTS)



/******************************************************************************
 * SRTS Control Structures
 ******************************************************************************/


/**
 * Value-type structure denoting keys-only sorting
 */
struct KeysOnlyType {};

/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <typename V>
__forceinline__ __host__ __device__ bool IsKeysOnly() {return false;}

/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <>
__forceinline__ __host__ __device__ bool IsKeysOnly<KeysOnlyType>() {return true;}

/**
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * tile_elements greater than the normal, and the last workload 
 * does the extra (problem-size % tile_elements) work.
 */
struct CtaDecomposition {
	int num_big_blocks;
	int big_block_elements;
	int normal_block_elements;
	int extra_elements_last_block;
	int num_elements;
};


/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <typename T, int BIT_START, int NUM_BITS> 
struct ExtractKeyBits 
{
	__device__ __forceinline__ static void Extract(int &bits, const T &source) 
	{
#if __CUDA_ARCH__ >= 200
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(BIT_START), "r"(NUM_BITS));
#else 
		const T MASK = (1 << NUM_BITS) - 1;
		bits = (source >> BIT_START) & MASK;
#endif
	}
};
	
/**
 * Extracts a bit field from source and places the zero or sign-extended result 
 * in extract
 */
template <int BIT_START, int NUM_BITS> 
struct ExtractKeyBits<unsigned long long, BIT_START, NUM_BITS> 
{
	__device__ __forceinline__ static void Extract(int &bits, const unsigned long long &source) 
	{
		const unsigned long long MASK = (1 << NUM_BITS) - 1;
		bits = (source >> BIT_START) & MASK;
	}
};
	



} // namespace b40c

