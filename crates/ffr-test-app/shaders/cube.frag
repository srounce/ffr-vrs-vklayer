#version 450

layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    // Heavy per-fragment ALU so the shading cost dominates and VRS savings are
    // visible in GPU timing.
    vec3 c = v_color;
    for (int i = 0; i < 160; ++i) {
        c = fract(sin(c * 12.9898 + float(i)) * 43758.5453) * 0.5 + v_color * 0.5;
    }
    out_color = vec4(mix(v_color, c, 0.15), 1.0);
}
