pub fn create_silly_payload() -> proto_data::data::SillyPayload {
  log::warn!("create_silly_payload");
  proto_data::data::SillyPayload {
    silly: 10,
  }
}