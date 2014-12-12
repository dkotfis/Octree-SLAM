// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
using namespace std;

#define SHOWBODY 0
#define SHOWLINES 0
#define SHOWVERTICES 0

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;
triangle* primitives2;
glm::vec3 lightDir;
bmp_texture* device_tex;

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		depthbuffer[index] = frag;
	}
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		return depthbuffer[index];
	}else{
		fragment f;
		return f;
	}
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		framebuffer[index] = value;
	}
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
	if(x<resolution.x && y<resolution.y){
		int index = (y*resolution.x) + x;
		return framebuffer[index];
	}else{
		return glm::vec3(0,0,0);
	}
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		image[index] = color;
	}
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		fragment f = frag;
		f.position.x = x;
		f.position.y = y;
		buffer[index] = f;
	}
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;      
		color.x = image[index].x*255.0;
		color.y = image[index].y*255.0;
		color.z = image[index].z*255.0;

		if(color.x>255){
			color.x = 255;
		}

		if(color.y>255){
			color.y = 255;
		}

		if(color.z>255){
			color.z = 255;
		}

		// Each thread writes one pixel location in the texture (textel)
		PBOpos[index].w = 0;
		PBOpos[index].x = color.x;     
		PBOpos[index].y = color.y;
		PBOpos[index].z = color.z;
	}
}

__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, glm::vec2 resolution, float zNear, float zFar, glm::mat4 projection, glm::mat4 view){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3){
		int idxX = 3*index;
		int idxY = idxX+1;
		int idxZ = idxX+2;

		//Transform position
		glm::vec4 pos = glm::vec4(vbo[idxX],vbo[idxY],vbo[idxZ],1.0f);
		glm::mat4 transformationMatrix = projection * view * glm::mat4();
		pos = transformationMatrix * pos;
		glm::vec4 new_pos = pos/pos.w;
		vbo[idxX] = -new_pos.x;
		vbo[idxY] = new_pos.y;
		vbo[idxZ] = new_pos.z;

		//Transform vertices
		vbo[idxX] = (resolution.x/2.0f) * (vbo[idxX]+1.0f);
		vbo[idxY] = (resolution.y/2.0f)  * (vbo[idxY]+1.0f);
		vbo[idxZ] = ((zFar - zNear)/2.0f) * (vbo[idxZ]+1.0f);

		//Transform normal
		glm::vec4 originnormal = glm::vec4(nbo[idxX],nbo[idxY],nbo[idxZ],0.0f);
		//originnormal = projection * originnormal;
		nbo[idxX] = originnormal.x;
		nbo[idxY] = originnormal.y;
		nbo[idxZ] = originnormal.z;
	}
}

__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount){
		int v0 = ibo[index*3];
		int v1 = ibo[index*3+1];
		int v2 = ibo[index*3+2];

		glm::vec3 c0(cbo[0], cbo[1], cbo[2]);
		glm::vec3 c1(cbo[3], cbo[4], cbo[5]);
		glm::vec3 c2(cbo[6], cbo[7], cbo[8]);

		glm::vec3 p0(vbo[v0*3], vbo[v0*3+1], vbo[v0*3+2]);
		glm::vec3 p1(vbo[v1*3], vbo[v1*3+1], vbo[v1*3+2]);
		glm::vec3 p2(vbo[v2*3], vbo[v2*3+1], vbo[v2*3+2]);

		glm::vec3 n0(nbo[v0*3], nbo[v0*3+1], nbo[v0*3+2]);
		glm::vec3 n1(nbo[v1*3], nbo[v1*3+1], nbo[v1*3+2]);
		glm::vec3 n2(nbo[v2*3], nbo[v2*3+1], nbo[v2*3+2]);

		primitives[index].c0 = c0;
		primitives[index].c1 = c1;
		primitives[index].c2 = c2;
		primitives[index].n0 = n0;
		primitives[index].n1 = n1;
		primitives[index].n2 = n2;
		primitives[index].p0 = p0;
		primitives[index].p1 = p1;
		primitives[index].p2 = p2;

	}
}

//Thrust predicate for triangle removal of backfacing
struct check_triangle {
  __host__ __device__
    bool operator() (const triangle& t) {
    float x1 = t.p1.x - t.p0.x;
    float y1 = t.p1.y - t.p0.y;
    float x2 = t.p2.x - t.p0.x;
    float y2 = t.p2.y - t.p0.y;

    return ((x1*y2 - y1*x2) < 0.0f);
  }
};

//Kernel to trim primitives before rasterization
__host__ void culling(triangle* primitives, triangle* new_primitives, int& numPrimitives) {
  thrust::device_ptr<triangle> in = thrust::device_pointer_cast<triangle>(primitives);
  thrust::device_ptr<triangle> out = thrust::device_pointer_cast<triangle>(new_primitives);
  numPrimitives = thrust::copy_if(in, in + numPrimitives, out, check_triangle()) - out;
}

__device__ glm::vec2 scanLineTriangleIntersect(glm::vec2 p1,glm::vec2 p2,glm::vec2 q1,glm::vec2 q2,glm::vec2 q3) {
	float min_t = 1.0f, max_t = 0.0f;
	glm::vec2 scanLine = p2 - p1;
	glm::vec2 triLine1 = q2 - q1;
	glm::vec2 triLine2 = q3 - q2;
	glm::vec2 triLine3 = q1 - q3;
	float crossValue[3]={0};

	//-------------------------------
	glm::vec2 cutLine = q1 - p1;
	crossValue[0] = scanLine.x*triLine1.y - scanLine.y*triLine1.x;
	crossValue[1] = cutLine.x*triLine1.y - triLine1.x*cutLine.y;
	crossValue[2] = cutLine.x*scanLine.y - scanLine.x*cutLine.y;
	if(abs(crossValue[0]) > 0.0001){
		float t = crossValue[1]/crossValue[0];
		float u = crossValue[2]/crossValue[0];
		if(u>0 && u<1 && t>0 && t<1){
			min_t = glm::min(t, min_t);
			max_t = glm::max(t, max_t);
		}
	}

	cutLine = q2 - p1;
	crossValue[0] = scanLine.x*triLine2.y - scanLine.y*triLine2.x;
	crossValue[1] = cutLine.x*triLine2.y - triLine2.x*cutLine.y;
	crossValue[2] = cutLine.x*scanLine.y - scanLine.x*cutLine.y;
	if(abs(crossValue[0]) > 0.0001){
		float t = crossValue[1]/crossValue[0];
		float u = crossValue[2]/crossValue[0];
		if(u>0 && u<1 && t>0 && t<1){
			min_t = glm::min(t, min_t);
			max_t = glm::max(t, max_t);
		}
	}

	cutLine = q3 - p1;
	crossValue[0] = scanLine.x*triLine3.y - scanLine.y*triLine3.x;
	crossValue[1] = cutLine.x*triLine3.y - triLine3.x*cutLine.y;
	crossValue[2] = cutLine.x*scanLine.y - scanLine.x*cutLine.y;
	if(abs(crossValue[0]) > 0.0001){
		float t = crossValue[1]/crossValue[0];
		float u = crossValue[2]/crossValue[0];
		if(u>0 && u<1 && t>0 && t<1){
			min_t = glm::min(t, min_t);
			max_t = glm::max(t, max_t);
		}
	}

	return glm::vec2(min_t, max_t);
}

//Interpolation by Barycentric Coordinates
//reference...http://mathworld.wolfram.com/BarycentricCoordinates.html
__device__ glm::vec3 bcInterpolate(glm::vec3 BC, glm::vec3 e1,glm::vec3 e2,glm::vec3 e3) {
	return BC.x * e1+ BC.y * e2 + BC.z * e3;
}

__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, float zNear, float zFar, bool barycenter){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		glm::vec3 primMin, primMax;
		triangle thisTri = primitives[index];
		getAABBForTriangle(thisTri, primMin, primMax);

		fragment frag;
		frag.primIndex = index;
		for(int j=glm::max(primMin.y,0.0f); j<glm::min(primMax.y,resolution.y)+1; j++){//scan from the bottom to the top, find left and right intersect points with triangle and draw it
			glm::vec2 portion = scanLineTriangleIntersect(glm::vec2(primMin.x, float(j)),glm::vec2(primMax.x, float(j)),glm::vec2(thisTri.p0),glm::vec2(thisTri.p1),glm::vec2(thisTri.p2));
			int leftX = primMin.x + portion.x * (primMax.x - primMin.x);
			int rightX = primMin.x + portion.y * (primMax.x - primMin.x) + 1; //have to add 1 for the last X

			for(int i=glm::max(leftX,0);i<glm::min(rightX,(int)resolution.x);i++){
				int screenPointIndex = (resolution.y - (j+1)) * resolution.x + (i-1);
				glm::vec3 bary = calculateBarycentricCoordinate (thisTri, glm::vec2(i,j));

				if (isBarycentricCoordInBounds(bary)){ //barycenter triangle interpolation
					frag.position.x = i;
					frag.position.y = j;
					frag.position.z = getZAtCoordinate(bary, thisTri);
					if(barycenter){
						frag.color =  bcInterpolate(bary, thisTri.c0, thisTri.c1, thisTri.c2);	
						//normal as color: (for normal value test)
						//frag.color = bcInterpolate(bary, thisTri.n0, thisTri.n1, thisTri.n2);
						frag.normal = bcInterpolate(bary, thisTri.n0, thisTri.n1, thisTri.n2);
						frag.normal = glm::normalize(frag.normal);
					}
					else{
						frag.color =  (thisTri.c0+thisTri.c1+thisTri.c2)/3.0f;
						frag.normal = (thisTri.n0+thisTri.n1+thisTri.n2)/3.0f;
						frag.normal = glm::normalize(frag.normal);
					}
					//show the most front primitive
					if (depthbuffer[screenPointIndex].position.z<frag.position.z && frag.position.z<-zNear && frag.position.z>-zFar){
						depthbuffer[screenPointIndex] = frag;
						depthbuffer[screenPointIndex].depth = 1;
					}
				}
			}
		}
	}

}


//Show Lines
__global__ void linesRasterizeKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, float zNear, float zFar){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		glm::vec3 primMin, primMax;
		triangle thisTri = primitives[index];
		getAABBForTriangle(thisTri, primMin, primMax);
		fragment frag;

		if(primMin.x>resolution.x || primMin.y>resolution.y || primMin.z>zFar || primMax.x<0 || primMax.y<0 ||primMax.z<zNear) //prim outside of the screen
			return;
		else{
			int startY = glm::max(int(primMin.y), 0);
			int endY = glm::min(int(primMax.y), (int)resolution.y);
			int startX = glm::max(int(primMin.x), 0);
			int endX = glm::min(int(primMax.x), (int)resolution.x);
			for(int j=startY; j<endY; j++){
				glm::vec2 intersectPortion = scanLineTriangleIntersect(glm::vec2(startX, float(j)),glm::vec2(endX, float(j)),glm::vec2(thisTri.p0),glm::vec2(thisTri.p1),glm::vec2(thisTri.p2));
				float left = intersectPortion.x;
				float right = intersectPortion.y;

				int leftstartX = startX + (endX-startX)*(float)left;
				int rightendX = startX + (endX-startX)*(float)right;

				for(int i=leftstartX; i<rightendX; i++){
					int screenPointIndex = ((resolution.y - j -1)*resolution.x) + i - 1;	
					if(i>0 && j>0 && i<resolution.x && j<resolution.y){

						glm::vec3 baryCoord = calculateBarycentricCoordinate (thisTri, glm::vec2 (i,j));
						if (!isBarycentricCoordInBounds(baryCoord)){ //show lines
							// Interpolation by BC	
							frag.position = bcInterpolate(baryCoord,thisTri.p0,thisTri.p1,thisTri.p2);	
							frag.position.z = -frag.position.z;
							frag.color =  glm::vec3(0,1,0);
							frag.normal = glm::vec3(0,0,1);

							//show the most front primitive
							if (frag.position.z<-zNear&&frag.position.z>-zFar)
								depthbuffer[screenPointIndex] = frag;
						}
					}
				}
			}
		}
	}
}


//Show Vertices
__global__ void verticesRasterizeKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, float zNear, float zFar){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		glm::vec3 point;
		//for each primitive, get each point, assign color and normal to the buffer
		for(int k=0; k<3; k++){
			if(k==0) point = primitives[index].p0;
			else if(k==1) point = primitives[index].p1;
			else point = primitives[index].p2;
			int i = glm::round(point.x);
			int j = glm::round(point.y);
			if(i>0 && j>0 && i<resolution.x && j<resolution.y){
				int depthIndex = ((resolution.y - j -1)*resolution.x) + i - 1;	
				fragment frag;
				frag.position = glm::vec3(point.x, point.y, point.z);
				frag.normal = glm::vec3(0,0,1);
				frag.color = glm::vec3(1,1,1);
				if (frag.position.z>zNear&&frag.position.z<zFar)
					depthbuffer[depthIndex] = frag;
			}
		}
	}
}

__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 *lightDir, bmp_texture *tex, glm::vec3 *device_data){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	 if (depthbuffer[index].position.z > -10000.0f) {
	  float diffuse = glm::dot(-*lightDir,depthbuffer[index].normal);
	  diffuse = diffuse>0?diffuse:0;
	  int x = depthbuffer[index].texcoord.x * tex->width;
	  int y = depthbuffer[index].texcoord.y * tex->height;
	  glm::vec3 tex_color0 = device_data[y * tex->height + x];
	  glm::vec3 tex_color1 = device_data[y * tex->height + x+1];
	  glm::vec3 tex_color2 = device_data[(y+1) * tex->height + x];
	  glm::vec3 tex_color3 = device_data[(y+1) * tex->height + x+1];

	  float xx = depthbuffer[index].texcoord.x * tex->width - x;
	  float yy = depthbuffer[index].texcoord.y * tex->height - y;
	  glm::vec3 tex_color = (tex_color0 * (1-xx) + tex_color1 * xx) * (1-yy) + (tex_color2 * (1-xx) + tex_color3 * xx) * yy;
	  depthbuffer[index].color = tex_color*diffuse*0.9f+tex_color*0.1f;
	 }
  }
}

//Handy function for reflection
__host__ __device__ glm::vec3 reflect(glm::vec3 vec_in, glm::vec3 norm) {
  return (vec_in - 2.0f*glm::dot(vec_in, norm)*norm);
}

//Phong shader
__global__ void fragmentShadePhongKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightpos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if (x <= resolution.x && y <= resolution.y){
    if (depthbuffer[index].position.z > -10000.0f) {
      //Store the fragment info locally for accessibility
      glm::vec3 V = depthbuffer[index].position;
      glm::vec3 N = depthbuffer[index].normal;

      //Compute necessary vectors
      glm::vec3 L = glm::normalize(lightpos- V);
      glm::vec3 E = glm::normalize(-V);
      glm::vec3 R = glm::normalize(reflect(-L, N));

      //Shininess
      float specPow = 4.0f;

      //Green (TODO: read from material)
      glm::vec3 green(0.0f, 1.0f, 0.0f);

      //Compute lighting
      glm::vec3 ambient = 0.1f * green;
      glm::vec3 diffuse = 0.45f * clamp(glm::dot(N, L), 0.0f, 1.0f) * green;
      glm::vec3 specular = 0.45f * clamp(pow(max(glm::dot(R, E), 0.0f), specPow), 0.0f, 1.0f) * green;
      depthbuffer[index].color = ambient + diffuse + specular;
    }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){
		framebuffer[index] = depthbuffer[index].color;
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, glm::mat4 rotationM, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, bmp_texture *tex, vector<glm::vec4> *texcoord, glm::vec3 eye, glm::vec3 center, glm::mat4 view, glm::vec3 lightpos, int mode, bool barycenter){

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

	//set up framebuffer
	framebuffer = NULL;
	cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));

	//set up depthbuffer
	depthbuffer = NULL;
	cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

	//kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));

	fragment frag;
	frag.color = glm::vec3(0,0,0);
	frag.normal = glm::vec3(0,0,0);
	frag.position = glm::vec3(0,0,-10000);
	clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

	//------------------------------
	//memory stuff
	//------------------------------
	primitives = NULL;
	cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));
  primitives2 = NULL;
  cudaMalloc((void**)&primitives2, (ibosize/3)*sizeof(triangle));

	device_ibo = NULL;
	cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
	cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

	device_vbo = NULL;
	cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
	cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_cbo = NULL;
	cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
	cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_nbo = NULL;
	cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
	cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

	lightDir = glm::vec3(0,0,-1);
    lightDir = glm::normalize(lightDir);
    glm::vec3 *device_lightDir = NULL;
    cudaMalloc((void**)&device_lightDir, cbosize*sizeof(glm::vec3));
    cudaMemcpy( device_lightDir, &lightDir, cbosize*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	device_tex = NULL;
    cudaMalloc((void**)&device_tex, sizeof(bmp_texture));
    cudaMemcpy( device_tex, tex, sizeof(bmp_texture), cudaMemcpyHostToDevice);
    glm::vec3 *device_data = NULL;
    cudaMalloc((void**)&device_data, tex->width * tex->height *sizeof(glm::vec3));
    cudaMemcpy( device_data, tex->data, tex->width * tex->height *sizeof(glm::vec3), cudaMemcpyHostToDevice);

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

	glm::vec3 up(0, 1, 0);
	float fovy = 60;
	float zNear = 0.001;
	float zFar = 10000;
	glm::mat4 perspectiveM = glm::perspective(fovy, resolution.x/resolution.y, zNear, zFar);

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, resolution, zNear, zFar, perspectiveM, view);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives);

	cudaDeviceSynchronize();

	int numOfPrimitives = ibosize/3 ;

  //------------------------------
  //culling
  //------------------------------
  culling(primitives, primitives2, numOfPrimitives);
  primitiveBlocks = ceil(((float)numOfPrimitives) / ((float)tileSize));
  triangle* temp = primitives;
  primitives = primitives2;
  primitives2 = temp;

	//------------------------------
	//rasterization
	//------------------------------
	if(SHOWBODY || mode==0){
		rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution, zNear, zFar, barycenter);
		cudaDeviceSynchronize();
	}
	if(SHOWLINES || mode==1){
		linesRasterizeKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution, zNear, zFar);
		cudaDeviceSynchronize();
	}
	if(SHOWVERTICES || mode==2){
		verticesRasterizeKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution, zNear, zFar);
	}

	//------------------------------
	//fragment shader
	//------------------------------
	 fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lightDir, device_tex, device_data);
    //fragmentShadePhongKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightpos);

	cudaDeviceSynchronize();

	//------------------------------
	//write fragments to framebuffer
	//------------------------------
	render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();

	kernelCleanup();

	checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
	cudaFree( primitives );
  cudaFree( primitives2 );
	cudaFree( device_vbo );
	cudaFree( device_cbo );
	cudaFree( device_ibo );
  cudaFree( device_nbo );
	cudaFree( framebuffer );
	cudaFree( depthbuffer );
}

