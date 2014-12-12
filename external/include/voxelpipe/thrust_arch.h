#include <map>
#include <algorithm>

namespace thrust
{
  namespace detail
  {
    namespace device
    {
      namespace cuda
      {
        namespace arch
        {
          namespace detail
          {
            inline void checked_get_current_device_properties(cudaDeviceProp &properties)
            {
              int current_device = -1;

              cudaError_t error = cudaGetDevice(&current_device);

              if (error)
              {
                throw thrust::system_error(error, thrust::cuda_category());
              }

              if (current_device < 0)
                throw thrust::system_error(cudaErrorNoDevice, thrust::cuda_category());

              // cache the result of the introspection call because it is expensive
              static std::map<int, cudaDeviceProp> properties_map;

              // search the cache for the properties
              std::map<int, cudaDeviceProp>::const_iterator iter = properties_map.find(current_device);

              if (iter == properties_map.end())
              {
                // the properties weren't found, ask the runtime to generate them
                error = cudaGetDeviceProperties(&properties, current_device);

                if (error)
                {
                  throw thrust::system_error(error, thrust::cuda_category());
                }

                // insert the new entry
                properties_map[current_device] = properties;
              } // end if
              else
              {
                // use the cached value
                properties = iter->second;
              } // end else
            } // end checked_get_current_device_properties()

            template <typename KernelFunction>
            inline void checked_get_function_attributes(cudaFuncAttributes& attributes, KernelFunction kernel)
            {
              typedef void(*fun_ptr_type)();

              // cache the result of the introspection call because it is expensive
              // cache fun_ptr_type rather than KernelFunction to avoid problems with long names on MSVC 2005
              static std::map<fun_ptr_type, cudaFuncAttributes> attributes_map;

              fun_ptr_type fun_ptr = reinterpret_cast<fun_ptr_type>(kernel);

              // search the cache for the attributes
              typename std::map<fun_ptr_type, cudaFuncAttributes>::const_iterator iter = attributes_map.find(fun_ptr);

              if (iter == attributes_map.end())
              {
                // the attributes weren't found, ask the runtime to generate them
                cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);

                if (error)
                {
                  throw thrust::system_error(error, thrust::cuda_category());
                }

                // insert the new entry
                attributes_map[fun_ptr] = attributes;
              } // end if
              else
              {
                // use the cached value
                attributes = iter->second;
              } // end else
            } // end checked_get_function_attributes()

          }

          inline size_t num_multiprocessors(const cudaDeviceProp& properties)
          {
             return properties.multiProcessorCount;
          } // end num_multiprocessors() 

          inline size_t max_active_threads_per_multiprocessor(const cudaDeviceProp& properties)
          {
            // index this array by [major, minor] revision
            // \see NVIDIA_CUDA_Programming_Guide_3.0.pdf p 140
            static const size_t max_active_threads_by_compute_capability[3][4] = \
            {{     0, 0, 0, 0},
            { 768, 768, 1024, 1024 },
            { 1536, 1536, 1536, 1536 }};

            // produce valid results for new, unknown devices
            if (properties.major > 2 || properties.minor > 3)
              return max_active_threads_by_compute_capability[2][3];
            else
              return max_active_threads_by_compute_capability[properties.major][properties.minor];
          } // end max_active_threads_per_multiprocessor()

          inline size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp& properties,
            const cudaFuncAttributes& attributes,
            size_t CTA_SIZE,
            size_t dynamic_smem_bytes)
          {
            // Determine the maximum number of CTAs that can be run simultaneously per SM
            // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet
            const size_t regAllocationUnit = (properties.major < 2 && properties.minor < 2) ? 256 : 512; // in registers
            const size_t warpAllocationMultiple = 2;
            const size_t smemAllocationUnit = 512;                                                 // in bytes
            const size_t maxThreadsPerSM = max_active_threads_per_multiprocessor(properties);      // 768, 1024, etc.
            const size_t maxBlocksPerSM = 8;

            // Number of warps (round up to nearest whole multiple of warp size & warp allocation multiple)
            const size_t numWarps = thrust::detail::util::round_i(thrust::detail::util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

            // Number of regs is regs per thread times number of warps times warp size
            const size_t regsPerCTA = thrust::detail::util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);

            const size_t smemBytes = attributes.sharedSizeBytes + dynamic_smem_bytes;
            const size_t smemPerCTA = thrust::detail::util::round_i(smemBytes, smemAllocationUnit);

            const size_t ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA : maxBlocksPerSM;
            const size_t ctaLimitSMem = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;
            const size_t ctaLimitThreads = maxThreadsPerSM / CTA_SIZE;

            return std::min<size_t>(ctaLimitRegs, std::min<size_t>(ctaLimitSMem, std::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
          }

          template <typename KernelFunction>
          inline size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
          {
            cudaDeviceProp properties;
            detail::checked_get_current_device_properties(properties);
            cudaFuncAttributes attributes;
            detail::checked_get_function_attributes(attributes, kernel);
            return num_multiprocessors(properties) * max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
          }
        }
      }
    }
  }
}