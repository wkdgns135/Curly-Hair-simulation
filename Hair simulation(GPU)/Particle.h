#ifndef __PARTICLE__
#define __PARTICLE__

#pragma once
#include "vector_types.h"

typedef struct Frame {
	float3 aim, up, cross;
}Frame;

class Particle
{
public:
	//Particle normalize position
	float3		*n_position;
	float		*density;
	float		*saturation;

public:
	double		*r_length;
	float3		*position;
	float3		*velocity;
	float3		*force;
	float3		*s_position;
	float3		*r_position;
	float3		*r_s_position;
	float3		*s_velocity;
	Frame		*r_s_frame;
	Frame		*s_frame;
	float3		*t;
	float3		*d;
	float3		*R;
};


#endif // !__PARTICLE__