fn main() -> Result<(), Box<dyn std::error::Error>> {
  let folder_path = "src/proto_build";
  std::fs::create_dir_all(folder_path).ok();
  let mut config = prost_build::Config::new();
  config.out_dir(folder_path);
  config.compile_protos(
    &[
      "data.proto",
    ],
    &[
      "."
    ])?;
  Ok(())
}
