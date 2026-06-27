#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 tint;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 0) out vec3 v_color;

void main() {
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    // Position-based gradient so cube faces are visibly shaded.
    v_color = (in_pos + 0.5) * pc.tint.rgb;
}
