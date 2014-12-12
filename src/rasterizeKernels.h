// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include "utilities.h"
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
using namespace std;

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, glm::mat4 rotationM,float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, bmp_texture *tex, vector<glm::vec4> *texcoord, glm::vec3, glm::vec3, glm::mat4 view, glm::vec3 lightpos, int mode, bool barycenter);

#endif //RASTERIZEKERNEL_H
