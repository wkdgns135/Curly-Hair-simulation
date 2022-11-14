#ifndef __FILE_CONTROLLER_H__
#define __FILE_CONTROLLER_H__

#include "HairModel.h"

#pragma once
vector<vector<float3>>			read_hair_asc(const char *filename, int num_strands);
void							vector2arr(vector<vector<float3>> v, float3 *p);
void							print_strand(std::ostream &oss, size_t start, size_t end_num);
void							out_hair_asc(HairModel *hm, const char *filename, int filenum);
void							capture(int _width, int _height);

#endif