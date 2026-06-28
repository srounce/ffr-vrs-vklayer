#version 450
#extension GL_EXT_fragment_shading_rate : require

// False-color the actual applied fragment shading rate so the foveation
// pattern is directly visible: green = 1x1 (full), yellow = 2x2, red = 4x4.
layout(location = 0) out vec4 out_color;

void main() {
    int sr = gl_ShadingRateEXT;
    int w = ((sr & gl_ShadingRateFlag4HorizontalPixelsEXT) != 0) ? 4
          : ((sr & gl_ShadingRateFlag2HorizontalPixelsEXT) != 0) ? 2 : 1;
    int h = ((sr & gl_ShadingRateFlag4VerticalPixelsEXT) != 0) ? 4
          : ((sr & gl_ShadingRateFlag2VerticalPixelsEXT) != 0) ? 2 : 1;
    int cov = w * h;

    vec3 c;
    if (cov <= 1)      c = vec3(0.0, 0.6, 0.0); // 1x1 full — green
    else if (cov <= 2) c = vec3(0.5, 0.7, 0.0);
    else if (cov <= 4) c = vec3(0.85, 0.85, 0.0); // 2x2 — yellow
    else if (cov <= 8) c = vec3(0.95, 0.5, 0.0);
    else               c = vec3(0.9, 0.0, 0.0); // 4x4 — red
    out_color = vec4(c, 1.0);
}
