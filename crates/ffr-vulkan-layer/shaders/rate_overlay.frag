#version 450
#extension GL_EXT_fragment_shading_rate : require

// Layer-side overlay: false-color the actually-applied shading rate so the
// foveation pattern is visible on top of any game's render.
layout(location = 0) out vec4 out_color;

void main() {
    int sr = gl_ShadingRateEXT;
    int w = ((sr & gl_ShadingRateFlag4HorizontalPixelsEXT) != 0) ? 4
          : ((sr & gl_ShadingRateFlag2HorizontalPixelsEXT) != 0) ? 2 : 1;
    int h = ((sr & gl_ShadingRateFlag4VerticalPixelsEXT) != 0) ? 4
          : ((sr & gl_ShadingRateFlag2VerticalPixelsEXT) != 0) ? 2 : 1;
    int cov = w * h;

    vec3 c;
    if (cov <= 1)      c = vec3(0.0, 1.0, 0.0); // 1x1 full — green
    else if (cov <= 2) c = vec3(0.6, 1.0, 0.0);
    else if (cov <= 4) c = vec3(1.0, 1.0, 0.0); // 2x2 — yellow
    else if (cov <= 8) c = vec3(1.0, 0.5, 0.0);
    else               c = vec3(1.0, 0.0, 0.0); // 4x4 — red
    out_color = vec4(c, 0.25);
}
