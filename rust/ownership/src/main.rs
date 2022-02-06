#![allow(dead_code)]

fn ownership1() {
  let mut str1: String;
  {
    let mut str = String::from("Hello");
    str.push_str(" world!");
    str1 = str; // move

    // println!("Str: {str}");  // Error!
  }
  println!("Str: {str1}");
  {
    let mut str = String::from("Hello");
    str.push_str(" universe!");
    str1 = str.clone();
  }
  println!("Str: {str1}");
}

fn take_ownership(s: String) {
  println!("take_ownership s: {s}");
}

fn ownership2() {
  let str = String::from("Hello");
  take_ownership(str);
  // println!("Str: {str}");  // Error!
}

fn return_ownership() -> String {
  let str = String::from("Some ret value");
  str
}

fn transfer_ownership(str: String) -> String {
  str
}

fn calc_len(s: String) -> (String, usize) {
  let len = s.len();
  (s, len)
}

fn ownership3() {
  let s1 = return_ownership();
  println!("Str: {s1}");
  let s2 = String::from("Hello");
  println!("Str: {s2}");
  let s3 = transfer_ownership(s2);
  // println!("Str: {s2}");   // Error!
  println!("Str: {s3}");

  let (s3, len) = calc_len(s3);
  println!("Str of length {len}: {s3}");
}

fn calc_len_ref(s: &String) -> usize {
  // s.push_str("Test");  // Error!
  s.len()
}

fn reference1() {
  let s1 = String::from("Hello");
  let r1 = &s1;
  let len = calc_len_ref(&s1);
  println!("Length of {s1} is {len}");
  println!("Ref is: {r1}");
}

fn append_ref(s: &mut String) {
  s.push_str(" World!")
}

fn reference2() {
  let mut s = String::from("hello");
  append_ref(&mut s);
  let r1 = &s;
  println!("Ref is: {r1}");
  let mr1 = &mut s;
  // println!("Ref is: {r1}");  // Error!
  // let mr2 = &mut s;  // Error!
  // let r2 = &s;   // Error
  println!("Ref is: {mr1}");
}

// fn dangle1() -> &String {
//   let s = String::from("Hello!");
//   &s
// }

// fn dangle2(s: String) -> (String, &String) {
//   (s, &s)
// }

fn dangle3() {
  let r;
  {
    let x = 5;
    r = &x;

    println!("r: {}", r);
  }

  // println!("r: {}", r);
}

fn no_dangle() -> String {
  let s = String::from("hello");
  s
}

fn longest<'a>(
  x: &'a str,
  y: &'a str,
) -> &'a str {
  if x.len() > y.len() {
    x
  } else {
    y
  }
}

use std::fmt::Display;

fn longest_with_anounce<'a, T>(
  x: &'a str,
  y: &'a str,
  ann: T,
) -> &'a str
where
  T: Display,
{
  println!("Anouncement! {ann}");
  if x.len() > y.len() {
    x
  } else {
    y
  }
}

fn lifetime1() {
  let str1 = String::from("Universe");
  let str2 = "World";
  println!("Longest: {}", longest(&str1, str2));
  {
    let str3 = String::from("World");
    let result = longest(&str1, &str3);
    println!("Longest: {result}",);
  }
  // Error
  // println!("Longest: {result}");
  longest_with_anounce(str1.as_str(), str1.as_str(), 666);
}

#[derive(Debug)]
struct LifetimeStruct<'a> {
  part: &'a str,
}

impl<'a> LifetimeStruct<'a> {
  fn level(&self) -> i32 {
    2
  }

  fn anounce_and_ret_part(
    &self,
    anounce: &str,
  ) -> &str {
    println!("Achtung: {anounce}");
    self.part
  }
}

fn lifetime2() {
  let novel = String::from("After writing a lot of Rust code, the Rust team found that Rust programmers were entering the same lifetime annotations over and over in particular situations.");
  let first_word = novel.split(' ').next().unwrap();
  let s = LifetimeStruct { part: first_word };
  println!("Struct: {s:?}");
  s.anounce_and_ret_part("What is happening?");
}

fn main() {
  ownership1();
  ownership2();
  ownership3();

  reference1();
  reference2();

  no_dangle();

  lifetime1();
  lifetime2();
}
