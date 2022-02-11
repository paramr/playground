use wasm_bindgen::prelude::*;

pub fn set_panic_hook() {
  console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);
}

macro_rules! console_log {
  // Note that this is using the `log` function imported above during
  // `bare_bones`
  ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen(start)]
pub fn main() {
  wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub fn greet(name: &str) {
  shared::create_silly_payload();
  console_log!("Hello {}!", name);
}
