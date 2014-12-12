/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <voxelpipe/compact_ranges.h>
#include "thrust_arch.h"
#include <thrust/scan.h>

#define CTA_SIZE 512
#define CTA_H    (512/32)

namespace voxelpipe {
namespace compact {

__forceinline__ __device__ int scan_popc(bool p, int& popc, const int tidx, volatile int *red)
{
    const uint32 mask = __ballot( p );
    popc = __popc( mask );
    return __popc( mask << (32 - tidx) );
}

// intra-warp inclusive scan
__forceinline__ __device__ void scan_warp(int tidx, unsigned int limit, volatile int *red)
{
    const uint32 val = red[tidx];

    // pad initial segment with zeros
    red[tidx] = 0;
    red += 32;

    // Hillis-Steele scan
    red[tidx]  = val;
    red[tidx] += red[tidx-1];
    red[tidx] += red[tidx-2];
    red[tidx] += red[tidx-4];
    red[tidx] += red[tidx-8];
    red[tidx] += red[tidx-16];

    // propagate resullpv back
    red[tidx-32] = red[tidx];
}
__forceinline__ __device__ int scan_popc(bool valid, volatile int* sm_warp_popc)
{
    int idx  = threadIdx.x;
	int tidx = threadIdx.x & 31;
	int widx = threadIdx.x / 32;
	__shared__ volatile int sm_red[CTA_SIZE*2];
	volatile int *sm_warp_red  = sm_red + widx*64;

    int popc;
	int eidx = scan_popc(valid,popc,tidx,sm_warp_red);

    if (tidx == 0)
    	sm_warp_popc[widx] = popc;      // population count of this warp

	__syncthreads();					// wait until all warps have written wpopc to shared mem

    const unsigned int warpcount = CTA_H;

    //  - use 1 warp to sum over wpopc
    if (widx == 0)
        scan_warp( idx, warpcount, sm_warp_popc );

    __syncthreads();

    return eidx;
}

// count the amount of output non-empty ranges in the source list
__global__ void compact_ranges_count(
          uint32*   offsets,
    const int32*    src_begin,
    const int32*    src_end,
    const uint32    n_elements,
    const uint32    n_blocks,
    const uint32    n_elements_per_block)
{
	//----------------------------------------------
	// Init
	//----------------------------------------------

	// useful variables (assumes 1D indexing)
	__shared__ volatile int sm_warp_popc[64];

    const uint32 block_id = blockIdx.x;
    const uint32 group_size = CTA_SIZE;

    const uint32 block_begin = block_id * n_elements_per_block;       // constant across CTA
    const uint32 block_end   = block_begin + n_elements_per_block;    // constant across CTA

    uint32 offset = 0;
    for (uint32 group_begin = block_begin; group_begin < block_end; group_begin += group_size)
    {
        const uint32 group_end = min( group_begin + group_size, n_elements );    // constant across CTA
        if (group_end <= group_begin)
            break;

        __syncthreads();

        //----------------------------------------------
	    // Compaction condition
	    //----------------------------------------------

        const uint32 local_id = threadIdx.x;
        const uint32 global_id = group_begin + local_id;

        // check if input should go to output
        bool valid = false;
        if (global_id < n_elements)
        {
            if (src_begin[ global_id ] < src_end[ global_id ])
                valid = true;
        }

	    //---------------------------------------------------
	    // Do an intra-cta reduction on the number of outputs
	    //---------------------------------------------------
        scan_popc( valid, sm_warp_popc );

        // ----------------------------------------------
        // Increment global offset
        // ----------------------------------------------
        const unsigned int warpcount = CTA_H;
        offset += sm_warp_popc[warpcount-1]; // constant across CTA

        __syncthreads();
    }
    if (threadIdx.x == 0)
        offsets[ block_id ] = offset;
}

// emit the compacted list of non-empty ranges
__global__ void compact_ranges_write(
    int32*          dest_begin,
    int32*          dest_end,
    int32*          dest_id,
    const uint32*   offsets,
    const int32*    src_begin,
    const int32*    src_end,
    const uint32    n_elements,
    const uint32    n_blocks,
    const uint32    n_elements_per_block)
{
	//----------------------------------------------
	// Init
	//----------------------------------------------

	// useful variables (assumes 1D indexing)
    const int widx = threadIdx.x / 32;
	__shared__ volatile int sm_warp_popc[64];

    const uint32 block_id = blockIdx.x;
    const uint32 group_size = CTA_SIZE;

    const uint32 block_begin = block_id * n_elements_per_block;       // constant across CTA
    const uint32 block_end   = block_begin + n_elements_per_block;    // constant across CTA

    uint32 offset = offsets[ block_id ]; // constant across CTA
    for (uint32 group_begin = block_begin; group_begin < block_end; group_begin += group_size)
    {
        const uint32 group_end = min( group_begin + group_size, n_elements );    // constant across CTA
        if (group_end <= group_begin)
            break;

        __syncthreads();

        //----------------------------------------------
	    // Compaction condition
	    //----------------------------------------------

        const uint32 local_id  = threadIdx.x;
        const uint32 global_id = group_begin + local_id;

        // check if input should go to output
        bool valid = false;
        int32 in_begin;
        int32 in_end;
        if (global_id < n_elements)
        {
            in_begin = src_begin[ global_id ];
            in_end   = src_end[ global_id ];
            if (in_begin < in_end)
                valid = true;
        }

	    //---------------------------------------------------
	    // Do an intra-cta reduction on the number of outputs
	    //---------------------------------------------------
        const int eidx = scan_popc( valid, sm_warp_popc );

	    //----------------------------------------------
	    // Write to compact output buffer
	    //----------------------------------------------
	    if (valid)
        {
            //const uint32 tpopc = (widx ? sm_warp_popc[widx-1] : 0u) + eidx;
            uint32 tpopc = eidx;
            if (widx)
                tpopc += sm_warp_popc[widx-1];
            const uint32 destIdx = offset + tpopc;

            dest_begin[destIdx] = in_begin;
            dest_end[destIdx]   = in_end;
            dest_id[destIdx]    = global_id;
        }

        // ----------------------------------------------
        // Increment global offset
        // ----------------------------------------------
        __syncthreads();

        const unsigned int warpcount = CTA_H;
        offset += sm_warp_popc[warpcount-1]; // constant across CTA
    }
}

} // namespace compact

Compact_ranges::Compact_ranges(const uint32 n)
{
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(compact::compact_ranges_count, CTA_SIZE, 0);

    m_counters.resize( max_blocks );
    m_offsets.resize( max_blocks );
}

// given two arrays {b[0], b[1], ..., b[n-1]} and
// {e[0], e[1], ..., e[n-1]} specifying a set of n
// possibly empty ranges { [b(i),e(i)) : i = 0,...,n-1 },
// return a copy of the two arrays with all the empty
// ranges removed, and an array specifying their position
// in the original list.
//
// \param dest_begin  output range start indices
// \param dest_end    output range end indices
// \param dest_id     output range index in the original list
// \param src_begin   input range start indices
// \param src_end     input range end indices
// \param n           number of input elements
// \result            number of output elements
//
uint32 Compact_ranges::run(int32* dest_begin, int32* dest_end, int32* dest_id, const int32* src_begin, const int32* src_end, const uint32 n_elements)
{
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(compact::compact_ranges_count, CTA_SIZE, 0);
    const uint32 group_size = CTA_SIZE;
    const uint32 n_groups   = (n_elements + group_size-1) / group_size;
    const size_t n_blocks   = std::min( (int)max_blocks, (int)n_groups );

    const uint32 n_groups_per_block   = (n_groups + n_blocks-1) / n_blocks;                     // constant across CTA
    const uint32 n_elements_per_block = n_groups_per_block * group_size;                        // constant across CTA

    uint32* counters_ptr = thrust::raw_pointer_cast( &*(m_counters.begin()) );
    uint32* offsets_ptr = thrust::raw_pointer_cast( &*(m_offsets.begin()) );

    // count the number of outputs per block
    compact::compact_ranges_count<<<n_blocks,CTA_SIZE>>>( counters_ptr, src_begin, src_end, n_elements, n_blocks, n_elements_per_block );

    cudaThreadSynchronize();

    // read the last block counter before it's overwritten
    const uint32 last_block = m_counters[n_blocks-1];

    // do an exclusive scan on the block counters to get proper offsets
    thrust::exclusive_scan(
        m_counters.begin(),
        m_counters.begin() + n_blocks,
        m_offsets.begin(),
        uint32(0) );

    cudaThreadSynchronize();

    // perform the actual writing
    compact::compact_ranges_write<<<n_blocks,CTA_SIZE>>>( dest_begin, dest_end, dest_id, offsets_ptr, src_begin, src_end, n_elements, n_blocks, n_elements_per_block  );

    cudaThreadSynchronize();

    // return number of output elements
    return m_offsets[n_blocks-1] + last_block;
}

#undef CTA_SIZE
#undef CTA_H

} // namespace voxelpipe

