#![allow(dead_code)]
#![allow(unused_variables)]

fn cause_panic() {
  panic!("Panic!");
}

fn index_out_of_bound_panic() {
  let v = vec![1, 2, 3];
  v[99];
}

use std::fs::File;
use std::io::{self, Read, ErrorKind};

fn result_match_usage() {
  let file_path = "target/hello.txt";
  let f = File::open(file_path);
  let f = match f {
    Ok(file) => file,
    Err(error) => match error.kind() {
      ErrorKind::NotFound => match File::create(file_path) {
        Ok(fc) => fc,
        Err(e) => panic!("Problem creating file: {e:?}"),
      }
      _ => panic!("Problem opening file: {error:?}")
    },
  };
}

fn result_unwrap_usage() {
  let file_path = "target/hello.txt";
  let f = File::open(file_path).unwrap_or_else(|error| {
    if error.kind() == ErrorKind::NotFound {
      File::create(file_path).unwrap_or_else(|error| {
        panic!("Problem creating file: {error:?}")
      })
    } else {
      panic!("Problem opening the file: {error:?}")
    }
  });
}

fn result_panic_usage() {
  let file_name = "target/hello.txt";
  // let f = File::open(file_name).unwrap();
  let f = File::open(file_name).expect("File not found");
}

fn error_propagation_match_read_file() -> Result<String, io::Error> {
  let f = File::open("target/hello.txt");
  let mut f = match f {
    Ok(file) => file,
    Err(e) => return Err(e),
  };
  let mut s = String::new();
  match f.read_to_string(&mut s) {
    Ok(_) => Ok(s),
    Err(e) => Err(e),
  }
}

#[derive(Debug)]
struct MyError {
  message: String,
}

impl From<io::Error> for MyError {
  fn from(err: io::Error) -> MyError {
    MyError { message: format!("MyError: {err}") }
  }
}

fn error_propagation_op_read_file() -> Result<String, MyError> {
  let mut s = String::new();
  let mut f = File::open("target/hello.txt")?;
  f.read_to_string(&mut s)?;
  Ok(s)
}

fn error_eat_op_read_file() -> Option<String> {
  let mut s = String::new();
  let mut f = File::open("target/hello.txt").ok()?;
  f.read_to_string(&mut s).ok()?;
  Some(s)
}

use std::panic;

fn main() {
  let result = panic::catch_unwind(|| {
    cause_panic();
  });
  println!("cause_panic: {result:?}");

  let result = panic::catch_unwind(|| {
    index_out_of_bound_panic();
  });
  println!("index_out_of_bound_panic: {result:?}");

  let result = panic::catch_unwind(|| {
    result_match_usage();
  });
  println!("result_match_usage: {result:?}");

  let result = panic::catch_unwind(|| {
    result_unwrap_usage();
  });
  println!("result_unwrap_usage: {result:?}");

  let result = panic::catch_unwind(|| {
    result_panic_usage();
  });
  println!("result_panic_usage: {result:?}");

  let result = error_propagation_match_read_file();
  println!("error_propagation_match_read_file: {result:?}");

  let result = error_propagation_op_read_file();
  println!("error_propagation_op_read_file: {result:?}");

  let result = error_eat_op_read_file();
  println!("error_eat_op_read_file: {result:?}");
}
