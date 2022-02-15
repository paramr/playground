fn main() -> Result<(), Box<dyn std::error::Error>> {
  let folder_path = "src/proto_build";
  std::fs::create_dir_all(folder_path).ok();
  let config = tonic_build::configure()
    .out_dir(folder_path)
    .build_client(false);
  config.compile(
      &[
        "server_api.proto",
      ],
      &[
        ".",
        "../data"
      ])?;
  Ok(())
}
