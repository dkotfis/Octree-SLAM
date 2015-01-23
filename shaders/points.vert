uniform mat4 u_mvpMatrix;

attribute vec3 v_position;
attribute vec3 v_color;

varying vec3 fs_color;

void main (void){
  fs_color = v_color;
  gl_Position = u_mvpMatrix * vec4(v_position, 1.0); 
}