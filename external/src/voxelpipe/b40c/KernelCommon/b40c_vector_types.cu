/**
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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 */

#pragma once

namespace b40c {

//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;


//
// Define general vector types
//

template <typename K> 
struct VecType<K, 1> {
	K x;
	typedef K Type;
};

template <typename K> 
struct VecType<K, 2> {
	K x;
	K y;
	typedef VecType<K, 2> Type;
};

template <typename K> 
struct VecType<K, 4> {
	K x;
	K y;
	K z;
	K w;
	typedef VecType<K, 4> Type;
};

//
// Specialize certain built-in vector types
//

#define B40C_DEFINE_VECTOR_TYPE(base_type,short_type)                           \
  template<> struct VecType<base_type, 1> { typedef short_type##1 Type; };      \
  template<> struct VecType<base_type, 2> { typedef short_type##2 Type; };      \
  template<> struct VecType<base_type, 4> { typedef short_type##4 Type; };     

B40C_DEFINE_VECTOR_TYPE(char,               char)
B40C_DEFINE_VECTOR_TYPE(signed char,        char)
B40C_DEFINE_VECTOR_TYPE(short,              short)
B40C_DEFINE_VECTOR_TYPE(int,                int)
B40C_DEFINE_VECTOR_TYPE(long,               long)
B40C_DEFINE_VECTOR_TYPE(long long,          longlong)
B40C_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
B40C_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
B40C_DEFINE_VECTOR_TYPE(unsigned int,       uint)
B40C_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
B40C_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
B40C_DEFINE_VECTOR_TYPE(float,              float)
B40C_DEFINE_VECTOR_TYPE(double,             double)

#undef B40C_DEFINE_VECTOR_TYPE


//------------------------------------------------------------------------------
// Utilities for moving types through global memory with cache modifiers
//------------------------------------------------------------------------------

enum CacheModifier {
	NONE,
	CG,
	CS
};


/**
 * Routines for modified loads through cache.  We use structs specialized by value 
 * type and cache-modifier to implement load operations
 */
template <typename T, CacheModifier CACHE_MODIFIER> struct GlobalLoad;

#if 0 // __CUDA_ARCH__ >= 300 

	/**
	 * Defines specialized load ops for only the base type 
	 */
	#define B40C_DEFINE_BASE_GLOBAL_LOAD(base_type, ptx_type, reg_mod)																												\
	template <> struct GlobalLoad<base_type, CG> {																												\
		__device__ __forceinline__ static void Ld(base_type &dest, base_type* d_ptr, int offset) {														\
			asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								\
	template <> struct GlobalLoad<base_type, CS> {																												\
		__device__ __forceinline__ static void Ld(base_type &dest, base_type* d_ptr, int offset) {														\
			asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
		}																																							\
	};																																								


	/**
	 * Defines specialized load ops for both the base type and for its derivative vector types
	 */
	#define B40C_DEFINE_GLOBAL_LOAD(base_type, dest_type, short_type, ptx_type, reg_mod)																												\
		template <> struct GlobalLoad<base_type, CG> {																												\
			__device__ __forceinline__ static void Ld(dest_type &dest, base_type* d_ptr, int offset) {														\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<base_type, CS> {																												\
			__device__ __forceinline__ static void Ld(dest_type &dest, base_type* d_ptr, int offset) {														\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##1, CG> {																												\
			__device__ __forceinline__ static void Ld(short_type##1 &dest, short_type##1* d_ptr, int offset) {														\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##1, CS> {																												\
			__device__ __forceinline__ static void Ld(short_type##1 &dest, short_type##1* d_ptr, int offset) {														\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(dest) : _B40C_ASM_PTR_(d_ptr + offset));																	\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##2, CG> {																												\
			__device__ __forceinline__ static void Ld(short_type##2 &dest, short_type##2* d_ptr, int offset) {													\
				asm("ld.global.cg.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y) : _B40C_ASM_PTR_(d_ptr + offset));										\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##2, CS> {																												\
			__device__ __forceinline__ static void Ld(short_type##2 &dest, short_type##2* d_ptr, int offset) {													\
				asm("ld.global.cs.v2."#ptx_type" {%0, %1}, [%2];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y) : _B40C_ASM_PTR_(d_ptr + offset));										\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##4, CG> {																												\
			__device__ __forceinline__ static void Ld(short_type##4 &dest, short_type##4* d_ptr, int offset) {													\
				asm("ld.global.cg.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y), "="#reg_mod(dest.z), "="#reg_mod(dest.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																							\
		};																																								\
		template <> struct GlobalLoad<short_type##4, CS> {																												\
			__device__ __forceinline__ static void Ld(short_type##4 &dest, short_type##4* d_ptr, int offset) {													\
				asm("ld.global.cs.v4."#ptx_type" {%0, %1, %2, %3}, [%4];" : "="#reg_mod(dest.x), "="#reg_mod(dest.y), "="#reg_mod(dest.z), "="#reg_mod(dest.w) : _B40C_ASM_PTR_(d_ptr + offset));	\
			}																																							\
		};

	// Cache-modified loads for built-in structures
	B40C_DEFINE_GLOBAL_LOAD(char, signed char, char, s8, h)
	B40C_DEFINE_BASE_GLOBAL_LOAD(signed char, s8, r)			// only need to define base: char2,char4, etc already defined from char
	B40C_DEFINE_GLOBAL_LOAD(short, short, short, s16, h)
	B40C_DEFINE_GLOBAL_LOAD(int, int, int, s32, r)
	B40C_DEFINE_GLOBAL_LOAD(long, long, long, s32, r)
	B40C_DEFINE_GLOBAL_LOAD(long long, long long, longlong, s64, l)
	B40C_DEFINE_GLOBAL_LOAD(unsigned char, unsigned char, uchar, u8, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned short, unsigned short, ushort, u16, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned int, unsigned int, uint, u32, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long, unsigned long, ulong, u32, r)
	B40C_DEFINE_GLOBAL_LOAD(unsigned long long, unsigned long long, ulonglong, u64, l)
	B40C_DEFINE_GLOBAL_LOAD(float, float, float, f32, f)
	B40C_DEFINE_BASE_GLOBAL_LOAD(double, f64, d)	// loads of vector-doubles don't compile
	
	#undef B40C_DEFINE_BASE_GLOBAL_LOAD
	#undef B40C_DEFINE_GLOBAL_LOAD

	// Workaround for the fact that the assembler reports an error when attempting to 
	// make vector loads of doubles.
	template <> struct GlobalLoad<double2, CG> {																												
		__device__ __forceinline__ static void Ld(double2 &dest, double2* d_ptr, int offset) {													
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.x) : _B40C_ASM_PTR_(d_ptr + offset));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.y) : _B40C_ASM_PTR_(d_ptr + offset + 1));																	
		}																																							
	};																																								
	template <> struct GlobalLoad<double4, CG> {																												
		__device__ __forceinline__ static void Ld(double4 &dest, double4* d_ptr, int offset) {													
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.x) : _B40C_ASM_PTR_(d_ptr + offset));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.y) : _B40C_ASM_PTR_(d_ptr + offset + 1));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.z) : _B40C_ASM_PTR_(d_ptr + offset + 2));																	
			asm("ld.global.cg.f64 %0, [%1];" : "=d"(dest.w) : _B40C_ASM_PTR_(d_ptr + offset + 3));																	
		}																																							
	};																																								
	
	// NONE-modified load 
	template <typename T> struct GlobalLoad<T, NONE>
	{
		__device__ __forceinline__ static void Ld(T &dest, T* d_ptr, int offset) {
			dest = d_ptr[offset]; 
		}
	};
	
	// NONE-modified load 
	template <> struct GlobalLoad<char, NONE>
	{
		__device__ __forceinline__ static void Ld(signed char &dest, char* d_ptr, int offset) {
			dest = d_ptr[offset]; 
		}
	};
	
#else 

	// Nothing is cached in these architectures: load normally
	template <typename T, CacheModifier CACHE_MODIFIER> struct GlobalLoad
	{
		__device__ __forceinline__ static void Ld(T &dest, T* d_ptr, int offset) {
			dest = d_ptr[offset]; 
		}
	};
	
	// Accomodate bizarre introduction of "signed" for char loads
	template <CacheModifier CACHE_MODIFIER> struct GlobalLoad<char, CACHE_MODIFIER>
	{
		__device__ __forceinline__ static void Ld(signed char &dest, char* d_ptr, int offset) {
			dest = d_ptr[offset]; 
		}
	};

#endif


	



} // namespace b40c

