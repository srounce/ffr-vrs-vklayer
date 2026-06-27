//! Compile the GLSL shaders to SPIR-V with glslangValidator at build time.

use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    for (src, spv) in [("shaders/cube.vert", "cube.vert.spv"), ("shaders/cube.frag", "cube.frag.spv")] {
        println!("cargo:rerun-if-changed={src}");
        let out = Path::new(&out_dir).join(spv);
        let status = Command::new("glslangValidator")
            .args(["-V", src, "-o"])
            .arg(&out)
            .status()
            .expect("failed to run glslangValidator (is glslang on PATH?)");
        assert!(status.success(), "shader compilation failed for {src}");
    }
}
