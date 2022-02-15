use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

fn naive_snake_case(name: Ident) -> String {
  let name = name.to_string();
  let mut s = String::new();
  let mut it = name.chars().peekable();
  while let Some(x) = it.next() {
      s.push(x.to_ascii_lowercase());
      if let Some(y) = it.peek() {
          if y.is_uppercase() {
              s.push('_');
          }
      }
  }
  s
}

struct WasmServiceGenerator {
  wasm_interface: TokenStream,
}

impl WasmServiceGenerator {
  fn new() -> Self {
    WasmServiceGenerator {
      wasm_interface: TokenStream::default(),
    }
  }

  fn generate_trait(
    service: &prost_build::Service,
    trait_name: Ident,
  ) -> TokenStream {
    let mut methods_tokens = TokenStream::new();
    for method in &service.methods {
      let name = quote::format_ident!("{}", method.name);
      if method.client_streaming || method.server_streaming {
        panic!("Client or server streaming not supported");
      }
      let req_message = quote::format_ident!("{}", method.input_type);
      let res_message = quote::format_ident!("{}", method.output_type);
      let method_tokens = quote! {
        fn #name(&mut self, request: #req_message) -> #res_message;
      };
      methods_tokens.extend(method_tokens);
    }
    quote! {
      pub trait #trait_name {
          #methods_tokens
      }
    }
  }

  fn generate_shim_impl(
    service: &prost_build::Service,
    trait_name: Ident,
  ) -> TokenStream {
    let shim_impl_name = quote::format_ident!("{}_impl", naive_snake_case(trait_name.clone()));
    let shim_impl_setter_name = quote::format_ident!("set_{}", naive_snake_case(shim_impl_name.clone()));
    let shim_impl_getter_name = quote::format_ident!("get_{}", naive_snake_case(shim_impl_name.clone()));
    let mut shim_impl_tokens = TokenStream::new();
    for method in &service.methods {
      let name = quote::format_ident!("{}", method.name);
      if method.client_streaming || method.server_streaming {
        panic!("Client or server streaming not supported");
      }
      let req_message = quote::format_ident!("{}", method.input_type);
      let method_impl_tokens = quote! {
        #[wasm_bindgen]
        pub fn #name(request_bytes: Vec<u8>) -> Vec<u8> {
          let request = #req_message::decode(&*request_bytes).expect("Error decoding!");
          let response = #shim_impl_getter_name().#name(request);
          let mut ret_bytes = Vec::new();
          response.encode(&mut ret_bytes).expect("Error encoding!");
          ret_bytes
        }
      };
      shim_impl_tokens.extend(method_impl_tokens);
    }
    quote! {
      use std::vec::Vec;
      use wasm_bindgen::prelude::wasm_bindgen;
      use prost::Message;
      #[allow(non_upper_case_globals)]
      static mut #shim_impl_name: Option<Box<dyn #trait_name>> = None;
      pub fn #shim_impl_setter_name(shim_impl: Box<dyn #trait_name>) {
        unsafe {
          #shim_impl_name = Some(shim_impl);
        }
      }
      #[inline(always)]
      pub fn #shim_impl_getter_name() -> &'static mut dyn #trait_name {
        unsafe {
          #shim_impl_name.as_mut().expect("setter for shim impl not called").as_mut()
        }
      }

      #shim_impl_tokens
    }
  }
}

impl prost_build::ServiceGenerator for WasmServiceGenerator {
  fn generate(&mut self, service: prost_build::Service, _buf: &mut String) {
    let trait_name = quote::format_ident!("{}", service.name);
    let trait_tokens = Self::generate_trait(&service, trait_name.clone());
    let shim_impl_tokens = Self::generate_shim_impl(&service, trait_name);
    self.wasm_interface = quote! {
      #trait_tokens

      #shim_impl_tokens
    };
  }

  fn finalize(&mut self, buf: &mut String) {
    println!("Parsing token stream: {}", self.wasm_interface.clone());
    let wasm_interface = std::mem::take(&mut self.wasm_interface);
    let ast: syn::File = syn::parse2(wasm_interface).expect("not a valid tokenstream");
    let code = prettyplease::unparse(&ast);
    buf.push_str(&code);
  }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let folder_path = "src/proto_build";
  std::fs::create_dir_all(folder_path).ok();
  let wasm_service_gen = Box::new(WasmServiceGenerator::new());
  let mut prost_build = prost_build::Config::new();
  prost_build.out_dir(folder_path);
  prost_build.service_generator(wasm_service_gen);
  prost_build.compile_protos(
    &[
      "wasm_api.proto",
    ],
    &[
      ".",
      "../../proto/data"
    ])?;
  Ok(())
}
