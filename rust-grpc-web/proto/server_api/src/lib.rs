pub mod api {
  include!("proto_build/server_api.rs");
}

pub mod data {
  // Reexport everything in proto_data here to compile server_api.
  pub use proto_data::data::*;
}