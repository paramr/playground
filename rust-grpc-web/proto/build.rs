fn main() -> Result<(), Box<dyn std::error::Error>> {
  tonic_build::configure()
    .compile(
      &[
        "shared.proto",
        "hello.proto",
      ],
      &[
        "."
      ])?;
  Ok(())
}
