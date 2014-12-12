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

#include <voxelpipe/b40c/KernelCommon/b40c_error_synchronize.cu>
#include <voxelpipe/b40c/LsbRadixSort/kernel/radixsort_reduction_kernel.cu>
#include <voxelpipe/b40c/LsbRadixSort/kernel/radixsort_spine_kernel.cu>
#include <voxelpipe/b40c/LsbRadixSort/kernel/radixsort_scanscatter_kernel.cu>

#include <voxelpipe/b40c/LsbRadixSort/radixsort_multi_cta.cu>

namespace b40c {


/**
 * Early-exit sorting enactor class.  
 * 
 * This sorting implementation is specifically designed for problems that are 
 * large enough to saturate the GPU (e.g., problems > 1M elements.)  
 * 
 * It also features "early-exit" digit-passes: when the sorting operation 
 * detects that all keys have the same digit at the same digit-place, the pass 
 * for that digit-place is short-circuited, reducing the cost of that pass 
 * by 80%.  This makes our implementation suitable for even low-degree binning 
 * problems where sorting would normally be overkill.  
 * 
 * To use, simply create a specialized instance of this class with your 
 * key-type K (and optionally value-type V if sorting with satellite 
 * values).  E.g., for sorting signed ints:
 * 
 * 		EarlyExitRadixSortingEnactor<int> sorting_enactor;
 * 
 * or for sorting floats paired with unsigned ints:
 * 			
 * 		EarlyExitRadixSortingEnactor<float, unsigned int> sorting_enactor;
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
template <typename K, typename V = KeysOnlyType, typename ConvertedKeyType = typename KeyConversion<K>::UnsignedBits>
class EarlyExitRadixSortingEnactor;



/**
 * Base class for early-exit, multi-CTA radix sorting enactors.
 */
template <typename K, typename V>
class BaseEarlyExitEnactor : public MultiCtaRadixSortingEnactor<K, V>
{
private:
	
	// Typedef for base class
	typedef MultiCtaRadixSortingEnactor<K, V> Base; 


protected:

	// Pair of "selector" device integers.  The first selects the incoming device 
	// vector for even passes, the second selects the odd.
	int *d_selectors;
	
	// Number of digit-place passes
	int passes;

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

		// Begin with device memory, subtract 192MB for video/spine/etc.  Factor in 
		// two vectors for keys (and values, if present)
		long long available_bytes = props.device_props.totalGlobalMem - (192 * 1024 * 1024);
		return available_bytes / (element_size * 2);
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

			// No override: figure it out
			
			if (props.device_sm_version < 120) {
				
				// G80/G90
				max_grid_size = props.device_props.multiProcessorCount * 4;
				
			} else if (props.device_sm_version < 200) {
				
				// GT200 
				max_grid_size = props.device_props.multiProcessorCount * 
						B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(props.kernel_ptx_version); 

				// Increase by default every 64 million key-values
				int step = 1024 * 1024 * 64;		 
				max_grid_size *= (MaxProblemSize(props) + step - 1) / step;
				
			} else {

				// GF100
				max_grid_size = 418;
			}
		} 
		
		return max_grid_size;
	}
	
	
protected:

	
	/**
	 * Constructor.
	 */
	BaseEarlyExitEnactor(
		int passes,
		int max_radix_bits,
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::MultiCtaRadixSortingEnactor(
				MaxGridSize(props, max_grid_size),
				B40C_RADIXSORT_TILE_ELEMENTS(props.kernel_ptx_version , ConvertedKeyType, V),
				max_radix_bits,
				props), 
			d_selectors(NULL),
			passes(passes)
	{
		// Allocate pair of ints denoting input and output vectors for even and odd passes
		cudaMalloc((void**) &d_selectors, 2 * sizeof(int));
	}

	
	
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

		int grid_size = num_elements / this->tile_elements;
		
		if (grid_size == 0) {

			// Always at least one block to process the remainder
			grid_size = 1;

		} else {
		
			if (this->cuda_props.device_sm_version == 130) {

				// GT200 Fine-tune sm130_clamped_grid_size to avoid CTA camping 

				int sm130_clamped_grid_size = this->cuda_props.device_props.multiProcessorCount * 
						B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(this->cuda_props.kernel_ptx_version); 

				// Increase by default every 64 million key-values
				int step = 1024 * 1024 * 64;		 
				sm130_clamped_grid_size *= (num_elements + step - 1) / step;

				double multiplier1 = 4.0;
				double multiplier2 = 16.0;

				double delta1 = 0.068;
				double delta2 = 0.127;	

				int dividend = (num_elements + this->tile_elements - 1) / this->tile_elements;

				int bumps = 0;
				while(true) {

					double quotient = ((double) dividend) / (multiplier1 * sm130_clamped_grid_size);
					quotient -= (int) quotient;

					if ((quotient > delta1) && (quotient < 1 - delta1)) {

						quotient = ((double) dividend) / (multiplier2 * sm130_clamped_grid_size / 3.0);
						quotient -= (int) quotient;

						if ((quotient > delta2) && (quotient < 1 - delta2)) {
							break;
						}
					}

					if (bumps == 3) {
						// Bump it down by 27
						sm130_clamped_grid_size -= 27;
						bumps = 0;
					} else {
						// Bump it down by 1
						sm130_clamped_grid_size--;
						bumps--;
					}
				}
				// Clamp to suggested grid size
				if (grid_size > sm130_clamped_grid_size) {
					grid_size = sm130_clamped_grid_size;
				}
			}
			
			// Clamp to mandated grid size
			if (grid_size > this->max_grid_size) {
				grid_size = this->max_grid_size;
			}
		}

		return grid_size;
	}
	

    /**
     * Post-sorting logic.
     */
    virtual cudaError_t PostSort(MultiCtaRadixSortStorage<K, V> &problem_storage, int passes) 
    {
    	// Copy out the selector from the last pass
    	int old_selector = problem_storage.selector;
    	
		cudaMemcpy(
			&problem_storage.selector, 
			&d_selectors[this->passes & 0x1], 
			sizeof(int), 
			cudaMemcpyDeviceToHost);
		
		problem_storage.selector ^= old_selector;
		
		return Base::PostSort(problem_storage, passes);
    }

    
    /**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
	cudaError_t DigitPlacePass(
		const int grid_size,
		const MultiCtaRadixSortStorage<K, V> &problem_storage, 
		const CtaDecomposition &work_decomposition)
	{
		// Compute number of spine elements to scan during this pass
		int spine_elements = grid_size * (1 << RADIX_BITS);
		int spine_tiles = (spine_elements + B40C_RADIXSORT_SPINE_TILE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		spine_elements = spine_tiles * B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		
		if (RADIXSORT_DEBUG && (PASS == 0)) {
    		
    		printf("\ndevice_sm_version: %d, kernel_ptx_version: %d\n", this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
    		printf("Bottom-level reduction & scan kernels:\n\tgrid_size: %d, \n\tthreads: %d, \n\ttile_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d\n\n",
    			grid_size, B40C_RADIXSORT_THREADS, this->tile_elements, work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
    		printf("Top-level spine scan:\n\tgrid_size: %d, \n\tthreads: %d, \n\tspine_block_elements: %d\n\n", 
    			grid_size, B40C_RADIXSORT_SPINE_THREADS, spine_elements);
    	}	

    	// Get kernel attributes
		cudaFuncAttributes reduce_kernel_attrs, spine_scan_kernel_attrs, scan_scatter_attrs;
		cudaFuncGetAttributes(
			&reduce_kernel_attrs, 
			LsbRakingReductionKernel<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor>);
		cudaFuncGetAttributes(
			&spine_scan_kernel_attrs, 
			LsbSpineScanKernel<void>);
		cudaFuncGetAttributes(
			&scan_scatter_attrs, 
			LsbScanScatterKernel<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor>);

		//
		// Counting Reduction
		//

		// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
		int dynamic_smem = (this->cuda_props.kernel_ptx_version >= 130) ? 
			scan_scatter_attrs.sharedSizeBytes - reduce_kernel_attrs.sharedSizeBytes : 
			0;

		LsbRakingReductionKernel<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor> <<<grid_size, B40C_RADIXSORT_THREADS, dynamic_smem>>>(
			d_selectors,
			this->d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[problem_storage.selector],
			(ConvertedKeyType *) problem_storage.d_keys[problem_storage.selector ^ 1],
			work_decomposition);
	    synchronize_if_enabled("RakingReduction");

		
		//
		// Spine
		//
		
		// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
		dynamic_smem = (this->cuda_props.kernel_ptx_version >= 130) ? 
			scan_scatter_attrs.sharedSizeBytes - spine_scan_kernel_attrs.sharedSizeBytes : 
			0;
		
		LsbSpineScanKernel<void><<<grid_size, B40C_RADIXSORT_SPINE_THREADS, dynamic_smem>>>(
			this->d_spine,
			this->d_spine,
			spine_elements);
	    synchronize_if_enabled("SrtsScanSpine");

		
		//
		// Scanning Scatter
		//

	    LsbScanScatterKernel<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<grid_size, B40C_RADIXSORT_THREADS, 0>>>(
			d_selectors,
			this->d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[problem_storage.selector],
			(ConvertedKeyType *) problem_storage.d_keys[problem_storage.selector ^ 1],
			problem_storage.d_values[problem_storage.selector],
			problem_storage.d_values[problem_storage.selector ^ 1],
			work_decomposition);
	    synchronize_if_enabled("ScanScatterDigits");

		return cudaSuccess;
	}
	
	
public:

	
    /**
     * Destructor
     */
    virtual ~BaseEarlyExitEnactor() 
    {
   		if (d_selectors) cudaFree(d_selectors);
    }
    
};




/******************************************************************************
 * Sorting enactor specializations
 ******************************************************************************/

/**
 * Sorting enactor that is specialized for for 8-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned char> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(2, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = this->GridSize(problem_storage.num_elements);
		this->GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		this->PreSort(problem_storage, 2);
		
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >(grid_size, problem_storage, work_decomposition); 

		this->PostSort(problem_storage, 2);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 16-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned short> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(4, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = this->GridSize(problem_storage.num_elements);
		this->GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		this->PreSort(problem_storage, 4);
		
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >(grid_size, problem_storage, work_decomposition); 

		this->PostSort(problem_storage, 4);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 32-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned int> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(8, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = this->GridSize(problem_storage.num_elements);
		this->GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		this->PreSort(problem_storage, 8);

		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<4, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<5, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<6, 4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<7, 4, 28, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (grid_size, problem_storage, work_decomposition); 

		this->PostSort(problem_storage, 8);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 64-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned long long> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(16, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(MultiCtaRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = this->GridSize(problem_storage.num_elements);
		this->GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		this->PreSort(problem_storage, 16);
		
		Base::template DigitPlacePass<0,  4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1,  4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2,  4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3,  4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<4,  4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<5,  4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<6,  4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<7,  4, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<8,  4, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<9,  4, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<10, 4, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<11, 4, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<12, 4, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<13, 4, 52, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<14, 4, 56, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<15, 4, 60, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (grid_size, problem_storage, work_decomposition); 

		this->PostSort(problem_storage, 16);

		return cudaSuccess;
	}
};


}// namespace b40c

