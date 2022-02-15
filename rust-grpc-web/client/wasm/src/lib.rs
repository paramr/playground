use wasm_bindgen::prelude::*;

pub mod wasm_api {
  include!("proto_build/wasm_api.rs");
}

pub mod data {
  // Reexport everything in proto_data here to compile wasm_api.
  pub use proto_data::data::*;
}

pub fn set_panic_hook() {
  console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
pub fn start() {
  wasm_logger::init(wasm_logger::Config::default());
  wasm_api::set_wasm_service_impl(Box::new(WasmServiceImpl::new()));
  log::info!("Wasm inited");
}

struct WasmServiceImpl {}

impl WasmServiceImpl {
  pub fn new() -> Self {
    WasmServiceImpl {}
  }
}

impl wasm_api::WasmService for WasmServiceImpl {
  fn hello(&mut self, request: wasm_api::WasmRequest) -> wasm_api::WasmResponse {
    wasm_api::WasmResponse {
      message: format!("Hello from wasm {:?}!", request.name).into(),
      payload: Some(shared::create_silly_payload()),
    }
  }

  fn hello_world(&mut self, request: wasm_api::WasmRequest) -> wasm_api::WasmResponse {
    wasm_api::WasmResponse {
      message: format!("Hello from wasm {:?}!", request.name).into(),
      payload: Some(shared::create_silly_payload()),
    }
  }
}