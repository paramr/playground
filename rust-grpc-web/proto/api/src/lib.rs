pub mod api {
  tonic::include_proto!("api");
}

pub mod data {
  // Reexport everything in proto_data here to compile api.
  pub use proto_data::data::*;
}