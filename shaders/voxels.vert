#version 420

uniform mat4 u_mvpMatrix;
uniform mat3 u_normMatrix;
uniform float u_scale;

out vec3 fs_position;
out vec3 fs_normal;
out vec3 fs_color;

layout (location = 0) in vec4 vox_cent;
layout (location = 1) in vec4 vox_color;

layout (binding = 0) uniform samplerBuffer voxel_centers;
layout (binding = 1) uniform samplerBuffer voxel_colors;

const vec3 cube_vert[8] = vec3[8](
	vec3(-1.0, -1.0, 1.0),
	vec3(1.0, -1.0, 1.0),
	vec3(1.0, 1.0, 1.0),
	vec3(-1.0, 1.0, 1.0),
	vec3(-1.0, -1.0, -1.0),
	vec3(1.0, -1.0, -1.0),
	vec3(1.0, 1.0, -1.0),
	vec3(-1.0, 1.0, -1.0)
);

const int cube_ind[36] = int[36] (
	0, 1, 2, 2, 3, 0, 
	3, 2, 6, 6, 7, 3, 
	7, 6, 5, 5, 4, 7, 
	4, 0, 3, 3, 7, 4, 
	0, 1, 5, 5, 4, 0,
	1, 5, 6, 6, 2, 1 
);

void main (void){
  gl_Position = u_mvpMatrix * vec4(cube_vert[cube_ind[gl_VertexID]]*u_scale + vec3(texelFetch(voxel_centers, gl_InstanceID)), 1.0);
  fs_position = vec3(gl_Position);
  fs_normal = u_normMatrix * normalize(cube_vert[cube_ind[gl_VertexID]]);
  fs_color = vec3(texelFetch(voxel_colors, gl_InstanceID));
}
