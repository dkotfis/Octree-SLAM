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

inline void synchronize(const char *message)
{
	cudaError_t error = cudaThreadSynchronize();
	if(error) {
		fprintf(stderr, "%s caused %d (%s)\n", message, error, cudaGetErrorString(error));
//		exit(1);
	}
} 

inline void synchronize_if_enabled(const char *message)
{
	// XXX this could potentially be a runtime decision
#if defined(__ERROR_SYNCHRONOUS)
	synchronize(message);
#endif
} 



} // namespace b40c

