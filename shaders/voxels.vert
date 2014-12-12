uniform mat4 u_mvpMatrix;
uniform mat4 u_projMatrix;
uniform mat3 u_normMatrix;

attribute vec3 v_position;
attribute vec3 v_normal;
attribute vec3 v_color;

varying vec3 fs_position;
varying vec3 fs_normal;
varying vec3 fs_color;

void main (void){
  fs_position = vec3(u_mvpMatrix * vec4(v_position, 1.0));
  fs_normal = u_normMatrix * v_normal;
  fs_color = v_color;
  gl_Position = u_mvpMatrix * vec4(v_position, 1.0);
}