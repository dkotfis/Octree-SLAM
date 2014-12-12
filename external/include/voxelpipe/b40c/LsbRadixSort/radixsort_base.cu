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

#include <voxelpipe/b40c/KernelCommon/b40c_kernel_utils.cu>
#include <voxelpipe/b40c/LsbRadixSort/kernel/radixsort_kernel_common.cu>


namespace b40c {


// Debugging options
static bool RADIXSORT_DEBUG = false;


/**
 * Class encapsulating device properties
 */
class CudaProperties 
{
public:
	
	// Information about our target device
	cudaDeviceProp 		device_props;
	int 				device_sm_version;
	
	// Information about our kernel assembly
	int 				kernel_ptx_version;
	
public:
	
	CudaProperties() 
	{
		// Get current device properties 
		int current_device;
		cudaGetDevice(&current_device);
		cudaGetDeviceProperties(&device_props, current_device);
		device_sm_version = device_props.major * 100 + device_props.minor * 10;
	
		// Get SM version of compiled kernel assemblies
		cudaFuncAttributes flush_kernel_attrs;
		cudaFuncGetAttributes(&flush_kernel_attrs, FlushKernel<void>);
		kernel_ptx_version = flush_kernel_attrs.ptxVersion * 10;
	}
	
};



/**
 * Base class for SRTS radix sorting enactors.
 */
template <typename K, typename V, typename Storage>
class BaseRadixSortingEnactor 
{
	
protected:

	/**
	 * Whether or not this instance can be used to sort satellite values
	 */
	static bool KeysOnly() 
	{
		return IsKeysOnly<V>();
	}

protected:

	//Device properties
	const CudaProperties cuda_props;
	
protected: 	
	
	/**
	 * Constructor.
	 */
	BaseRadixSortingEnactor(const CudaProperties &props = CudaProperties()) : 
		cuda_props(props) {}


public:
	

	/**
     * Destructor
     */
    virtual ~BaseRadixSortingEnactor() {}

    
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(Storage &problem_storage) = 0;	
    
};





}// namespace b40c

