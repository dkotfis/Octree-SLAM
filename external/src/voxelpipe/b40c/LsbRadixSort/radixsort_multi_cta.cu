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
 ******************************************************************************/

#pragma once

#include "radixsort_base.cu"

namespace b40c {


/**
 * Storage management structure for multi-CTA-sorting device vectors.
 * 
 * Multi-CTA sorting is performed out-of-core, meaning that sorting passes
 * must have two equally sized arrays: one for reading in from, the other for 
 * writing out to.  As such, this structure maintains a pair of device vectors 
 * for keys (and for values), and a "selector" member to index which vector 
 * contains valid data (i.e., the data to-be-sorted, or the valid-sorted data 
 * after a sorting operation). 
 * 
 * E.g., consider a MultiCtaRadixSortStorage "device_storage".  The valid data 
 * should always be accessible by: 
 * 
 * 		device_storage.d_keys[device_storage.selector];
 * 
 * The non-selected array(s) can be allocated lazily upon first sorting by the 
 * sorting enactor if left NULL, or a-priori by the caller.  (If user-allocated, 
 * they should be large enough to accomodate num_elements.)    
 * 
 * It is the caller's responsibility to free any non-NULL storage arrays when
 * no longer needed.  This allows for the storage to be re-used for subsequent 
 * sorting operations of the same size.
 * 
 * NOTE: After a sorting operation has completed, the selecter member will
 * index the key (and value) pointers that contain the final sorted results.
 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if 
 * the input started in d_keys[0].)
 * 
 */
template <typename K, typename V = KeysOnlyType> 
struct MultiCtaRadixSortStorage
{
	// Pair of device vector pointers for keys
	K* d_keys[2];
	
	// Pair of device vector pointers for values
	V* d_values[2];

	// Number of elements for sorting in the above vectors 
	int num_elements;
	
	// Selector into the pair of device vector pointers indicating valid 
	// sorting elements (i.e., where the results are)
	int selector;

    // Constructor
	MultiCtaRadixSortStorage(int num_elements = 0) :
		num_elements(num_elements), 
		selector(0) 
	{
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}

	// Constructor
	MultiCtaRadixSortStorage(int num_elements, K* keys, V* values = NULL) :
		num_elements(num_elements), 
		selector(0) 
	{
		d_keys[0] = keys;
		d_keys[1] = NULL;
		d_values[0] = values;
		d_values[1] = NULL;
	}
};


/**
 * Base class for multi-CTA sorting enactors
 */
template <typename K, typename V, typename Storage = MultiCtaRadixSortStorage<K, V> >
class MultiCtaRadixSortingEnactor : 
	public BaseRadixSortingEnactor<K, V, Storage>
{
private:
	
	typedef BaseRadixSortingEnactor<K, V, Storage> Base; 
	
protected:
	
	// Maximum number of threadblocks this enactor will launch
	int max_grid_size;

	// Fixed "tile size" of keys by which threadblocks iterate over 
	int tile_elements;
	
	// Temporary device storage needed for scanning digit histograms produced
	// by separate CTAs
	int *d_spine;
	
protected:
	
	/**
	 * Constructor.
	 */
	MultiCtaRadixSortingEnactor(
		int max_grid_size,
		int tile_elements,
		int max_radix_bits,
		const CudaProperties &props = CudaProperties()) : 
			Base::BaseRadixSortingEnactor(props),  
			max_grid_size(max_grid_size), 
			tile_elements(tile_elements),
			d_spine(NULL)
	{
		// Allocate the spine
		int spine_elements = max_grid_size * (1 << max_radix_bits);
		int spine_tiles = (spine_elements + B40C_RADIXSORT_SPINE_TILE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		spine_elements = spine_tiles * B40C_RADIXSORT_SPINE_TILE_ELEMENTS;
		cudaMalloc((void**) &d_spine, spine_elements * sizeof(int));
	}


	/**
	 * Computes the work-decomposition amongst CTAs for the give problem size 
	 * and grid size
	 */
	void GetWorkDecomposition(
		int num_elements, 
		int grid_size,
		CtaDecomposition &work_decomposition) 
	{
		int total_tiles 		= (num_elements + tile_elements - 1) / tile_elements;
		int tiles_per_block 	= total_tiles / grid_size;						
		int extra_tiles 		= total_tiles - (tiles_per_block * grid_size);

		work_decomposition.num_big_blocks 				= extra_tiles;
		work_decomposition.big_block_elements 			= (tiles_per_block + 1) * tile_elements;
		work_decomposition.normal_block_elements 		= tiles_per_block * tile_elements;
		work_decomposition.extra_elements_last_block 	= num_elements - (num_elements / tile_elements * tile_elements);
		work_decomposition.num_elements 				= num_elements;
	}
	
	
    /**
     * Pre-sorting logic.
     */
    virtual cudaError_t PreSort(Storage &problem_storage, int passes) 
    {
    	// Allocate device memory for temporary storage (if necessary)
    	if (problem_storage.d_keys[0] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[0], problem_storage.num_elements * sizeof(K));
    	}
    	if (problem_storage.d_keys[1] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[1], problem_storage.num_elements * sizeof(K));
    	}
    	if (!Base::KeysOnly()) {
    		if (problem_storage.d_values[0] == NULL) {
    			cudaMalloc((void**) &problem_storage.d_values[0], problem_storage.num_elements * sizeof(V));
    		}
    		if (problem_storage.d_values[1] == NULL) {
    			cudaMalloc((void**) &problem_storage.d_values[1], problem_storage.num_elements * sizeof(V));
    		}
    	}

    	return cudaSuccess;
    }
    
    
    /**
     * Post-sorting logic.
     */
    virtual cudaError_t PostSort(MultiCtaRadixSortStorage<K, V> &problem_storage, int passes) 
    {
    	return cudaSuccess;
    }	

    
public:

    /**
     * Destructor
     */
    virtual ~MultiCtaRadixSortingEnactor() 
    {
   		if (d_spine) cudaFree(d_spine);
    }
    
};






}// namespace b40c

