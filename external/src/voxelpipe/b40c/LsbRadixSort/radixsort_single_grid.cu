/******************************************************************************
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
 ******************************************************************************/



/******************************************************************************
 * Radix Sorting API
 *
 ******************************************************************************/

#pragma once

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <b40c/KernelCommon/b40c_error_synchronize.cu>
#include <b40c/LsbRadixSort/radixsort_multi_cta.cu>
#include <b40c/LsbRadixSort/kernel/radixsort_singlegrid_kernel.cu>

namespace b40c {

/**
 * Single-grid sorting enactor class.  
 * 
 * This sorting implementation is specifically designed for small problems 
 * that are not large enough to saturate the GPU (e.g., problems < 1M elements.)
 * It performs multiple digit-place passes over the input problem all within
 * a single kernel launch.  It does so by implementing software global-barriers
 * across threadblocks.  
 * 
 * NOTE: This enactor can NOT be used to sort:
 * 		- Problems having structured value-types (i.e., keys that are paired with 
 * 			satellite values that are structs or classes).  This is because
 * 			the compiler cannot be told how to copy structures from global 
 * 			memory using volatile or cache-global load modifiers. 
 * 
 * It also allows the caller to specify the lower-order bits over which the 
 * keys should be sorted (e.g., the lower 17 bits out of 32-bit keys).  This
 * reduces the number of overall sorting passes (4-bits per pass) for 
 * scenarios in which the keyspace can be restricted in this manner.  
 * 
 * To use, simply create a specialized instance of this class with your 
 * key-type K (and optionally value-type V if sorting with satellite 
 * values).  E.g., for sorting signed ints:
 * 
 * 		SingleGridRadixSortingEnactor<int> sorting_enactor;
 * 
 * or for sorting floats paired with unsigned ints:
 * 			
 * 		SingleGridRadixSortingEnactor<float, unsigned int> sorting_enactor;
 * 
 * The enactor itself manages a small amount of device state for use when 
 * performing sorting operations.  To minimize GPU allocation overhead, 
 * enactors can be re-used over multiple sorting operations.  
 * 
 * The problem-storage for a sorting operation is independent of the sorting
 * enactor.  A single enactor can be reused to sort multiple instances of the
 * same type of problem storage.  The MultiCtaRadixSortStorage structure
 * is used to manage the input/output/temporary buffers needed to sort 
 * a problem of a given size.  This enactor will lazily allocate any NULL
 * buffers contained within a problem-storage structure.  
 *
 * Sorting is invoked upon a problem-storage as follows:
 * 
 * 		sorting_enactor.EnactSort(device_storage);
 * 
 * or
 * 
 * 		sorting_enactor.EnactSort<17>(device_storage);
 * 
 * in the case where the caller knows that only the lower 17 bits
 * are used to differentiate keys.  N.B.: for use within templated
 * functions, the proper syntax is:
 *  
 * 		sorting_enactor.template EnactSort<17>(device_storage);
 * 
 * This enactor will update the selector within the problem storage
 * to indicate which buffer contains the sorted output. E.g., 
 * 
 * 		device_storage.d_keys[device_storage.selector];
 * 
 * Please see the overview of MultiCtaRadixSortStorage for more details.
 * 
 * 
 * @template-param K
 * 		Type of keys to be sorted
 * @template-param V
 * 		Type of values to be sorted.
 * @template-param ConvertedKeyType
 * 		Leave as default to effect necessary enactor specialization for 
 * 		signed and floating-point types
 */
template <typename K, typename V = KeysOnlyType>
class SingleGridRadixSortingEnactor;



/**
 * Template-specialized structure for invoking the single-grid kernel a 
 * specific number of times.  We extract this so as to avoid unnecessary
 * kernel generation.
 */
template<int INVOCATIONS, typename K, typename V, int RADIX_BITS, int PASSES> 
struct SingleGridKernelInvoker;


/**
 * Single-grid sorting enactor class.  
 */
template <typename K, typename V>
class SingleGridRadixSortingEnactor : public MultiCtaRadixSortingEnactor<K, V>
{
private:
	
	// Typedef for base class
	typedef MultiCtaRadixSortingEnactor<K, V> Base; 

protected:
	
	// Array of global synchronization counters, one for each threadblock
	int *d_sync;

	
public: 
	
	// Unsigned integer type suitable for radix sorting of keys
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

	
	/**
	 * Utility function: Returns the maximum problem size this enactor can sort on the device
	 * it was initialized for.
	 */
	static long long MaxProblemSize(const CudaProperties &props) 
	{
		long long element_size = (Base::KeysOnly()) ? sizeof(K) : sizeof(K) + sizeof(V);

		// Begin with device memory, subtract 128MB for video/spine/etc.  Factor in 
		// three vectors for keys (and values, if present)
		long long available_bytes = props.device_props.totalGlobalMem - 128;
		return available_bytes / (element_size * 3);
	}


protected:
	
	// Radix bits per pass
	static const int RADIX_BITS = 4;
	
	
	/**
	 * Utility function: Returns the default maximum number of threadblocks 
	 * this enactor class can launch.
	 */
	static int MaxGridSize(const CudaProperties &props, int max_grid_size = 0) 
	{
		if (max_grid_size == 0) {
			// No override: Fully populate all SMs
			max_grid_size = props.device_props.multiProcessorCount * 
					B40C_RADIXSORT_SG_OCCUPANCY(props.kernel_ptx_version); 
		} 
		return max_grid_size;
	}
	
	
protected:
	
	/**
	 * Determines the actual number of CTAs to launch for the given problem size
	 * 
	 * @return The actual number of CTAs that should be launched
	 */
	int GridSize(int num_elements)
	{
		// Initially assume that each threadblock will do only one 
		// tile worth of work (and that the last one will do any remainder), 
		// but then clamp it by the "max" restriction  

		int grid_size = (num_elements + this->tile_elements - 1) / this->tile_elements;
		
		if (grid_size == 0) {
			// Always at least one block to process the remainder
			grid_size = 1;
		} else if (grid_size > this->max_grid_size) {
			grid_size = this->max_grid_size;
		}
		
		return grid_size;
	}

	
	/**
     * Post-sorting logic.
     */
    virtual cudaError_t PostSort(MultiCtaRadixSortStorage<K, V> &problem_storage, int passes) 
    {
		problem_storage.selector = passes & 0x1;
		return Base::PostSort(problem_storage, passes);
    }

public:

	/**
	 * Constructor.
	 */
	SingleGridRadixSortingEnactor(
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::MultiCtaRadixSortingEnactor(
				MaxGridSize(props, max_grid_size),
				B40C_RADIXSORT_SG_TILE_ELEMENTS(props.kernel_ptx_version , ConvertedKeyType, V),
				RADIX_BITS,
				props), 
			d_sync(NULL)
	{
		// Allocate and initialize synchronization counters
		cudaMalloc((void**) &d_sync, sizeof(int) * this->max_grid_size);
		InitSync<void><<<this->max_grid_size, 32, 0>>>(d_sync);
	}

	
	/**
     * Destructor
     */
    virtual ~SingleGridRadixSortingEnactor() 
    {
    	if (d_sync) cudaFree(d_sync);
    }
    
	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <int LOWER_KEY_BITS>
	cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		const int PASSES = (LOWER_KEY_BITS + RADIX_BITS - 1) / RADIX_BITS;

		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = GridSize(problem_storage.num_elements);	
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);
		
		// Compute number of spine elements to scan during this pass
		int spine_elements = grid_size * (1 << RADIX_BITS);
		int spine_tiles = (spine_elements + B40C_RADIXSORT_SPINE_TILE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		spine_elements = spine_tiles * B40C_RADIXSORT_SPINE_TILE_ELEMENTS;

		// Perform any lazy allocation
		PreSort(problem_storage, PASSES);
		
		if (RADIXSORT_DEBUG) {
    		printf("\ndevice_sm_version: %d, kernel_ptx_version: %d\n", 
    			this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
    		printf("%d-bit bottom-level reduction & scan kernels:\n\tgrid_size: %d, \n\tthreads: %d, \n\ttile_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d\n\n",
    			LOWER_KEY_BITS, grid_size, B40C_RADIXSORT_THREADS, this->tile_elements, work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
    		printf("Top-level spine scan:\n\tgrid_size: %d, \n\tthreads: %d, \n\tspine_block_elements: %d\n\n", 
    			grid_size, B40C_RADIXSORT_SPINE_THREADS, spine_elements);
    	}	
		
		// Invoke kernel
		SingleGridKernelInvoker<(PASSES + 8 - 1) / 8, K, V, RADIX_BITS, PASSES>::Invoke(
			grid_size,	
			this->d_sync,
			this->d_spine, 
			problem_storage, 
			work_decomposition, 
			spine_elements);

	    // Perform any post-mortem
		PostSort(problem_storage, PASSES);

		return cudaSuccess;
	}
	
	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		return EnactSort<sizeof(K) * 8>(problem_storage);	// mooch
	}
	
};



/**
 * Template specialization for one invocation of the sorting kernel (which 
 * performs up to 8 passes).
 */
template<typename K, typename V, int RADIX_BITS, int PASSES> 
struct SingleGridKernelInvoker <1, K, V, RADIX_BITS, PASSES>
{
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

	static void Invoke(
		int grid_size,	
		int *d_sync,
		int *d_spine, 
		MultiCtaRadixSortStorage<K, V> &problem_storage, 
		CtaDecomposition &work_decomposition, 
		int spine_elements)
	{
		LsbSingleGridSortingKernel<ConvertedKeyType, V, RADIX_BITS, PASSES, 0, PreprocessKeyFunctor<K>, PostprocessKeyFunctor<K> ><<<grid_size, B40C_RADIXSORT_THREADS, 0>>>(
			d_sync,
			d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[0],
			(ConvertedKeyType *) problem_storage.d_keys[1],
			problem_storage.d_values[0],
			problem_storage.d_values[1],
			work_decomposition,
			spine_elements);
	    synchronize_if_enabled("ScanScatterDigits");
	}
};


/**
 * Template specialization for two invocations of the sorting kernel (which 
 * performs up to 8 passes).
 */
template<typename K, typename V, int RADIX_BITS, int PASSES> 
struct SingleGridKernelInvoker <2, K, V, RADIX_BITS, PASSES>
{
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

	static void Invoke(
		int grid_size,	
		int *d_sync,
		int *d_spine, 
		MultiCtaRadixSortStorage<K, V> &problem_storage, 
		CtaDecomposition &work_decomposition, 
		int spine_elements)
	{
		LsbSingleGridSortingKernel<ConvertedKeyType, V, RADIX_BITS, PASSES, 0, PreprocessKeyFunctor<K>, PostprocessKeyFunctor<K> ><<<grid_size, B40C_RADIXSORT_THREADS, 0>>>(
			d_sync,
			d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[0],
			(ConvertedKeyType *) problem_storage.d_keys[1],
			problem_storage.d_values[0],
			problem_storage.d_values[1],
			work_decomposition,
			spine_elements);
	    synchronize_if_enabled("ScanScatterDigits");

	    LsbSingleGridSortingKernel<ConvertedKeyType, V, RADIX_BITS, PASSES, 8, PreprocessKeyFunctor<K>, PostprocessKeyFunctor<K> ><<<grid_size, B40C_RADIXSORT_THREADS, 0>>>(
			d_sync,
			d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[0],
			(ConvertedKeyType *) problem_storage.d_keys[1],
			problem_storage.d_values[0],
			problem_storage.d_values[1],
			work_decomposition,
			spine_elements);
	    synchronize_if_enabled("ScanScatterDigits");
	}
};




}// namespace b40c

