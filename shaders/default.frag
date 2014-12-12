uniform vec3 u_light;

varying vec3 fs_position;
varying vec3 fs_normal;

void main (void){
   vec3 L = normalize(u_light - fs_position);
   vec3 E = normalize(-fs_position); // we are in Eye Coordinates, so EyePos is (0,0,0) 
   vec3 R = normalize(-reflect(L, fs_normal));

   //calculate Ambient Term:
   vec4 Iamb = vec4(0.0, 0.1, 0.0, 1.0);

   //calculate Diffuse Term:
   vec4 Idiff = vec4(0.0, 0.45, 0.0, 1.0) * max(dot(fs_normal,L), 0.0);
   Idiff = clamp(Idiff, 0.0, 1.0);

   // calculate Specular Term:
   vec4 Ispec = vec4(0.0, 0.45, 0.0, 1.0)
                * pow(max(dot(R, E), 0.0), 0.3 * 4.0);
   Ispec = clamp(Ispec, 0.0, 1.0);
   // write Total Color:
   gl_FragColor = Iamb + Idiff + Ispec;

}