﻿#pragma once
#include "HairModel.cuh"
#include "HairModel.h"
#include "vector_calc.h"


void cuda_init(Strand *s) {	
	cudaMemcpyToSymbol("s_d", s, sizeof(s));
}

void simulation() {

}