fn vec_create_test() {
  // type infered by insert :o
  let mut v = Vec::new();
  v.push(5);
  v.push(6);
  dbg!(v);

  let v = vec![1, 2, 3];
  dbg!(v);

  let mut v: Vec<String> = Vec::new();
  v.push(String::from("1"));
  v.push(String::from("2"));
  v.push(String::from("3"));
  dbg!(v);
}

fn vec_read_test() {
  let v = vec![1, 2, 3];
  let third_element: &i32 = &v[2];
  println!("The third element is {third_element}");

  match v.get(3) {
    Some(num) => println!("The element was: {num}"),
    None => println!("No element found"),
  }
}

#[allow(dead_code)]
fn vec_mutable_test() {
  let mut v = vec![1, 2, 3, 4, 5];
  #[allow(unused_variables)]
  let first = &v[0];
  v.push(6);
  // Uncommenting the line below causes error
  // because of mutating a vec with borrowed immutable
  // reference
  // println!("First element is: {first}");
}

fn vec_iter_test() {
  let mut v = vec![1, 2, 3];
  for i in &v {
    println!("{i}");
  }
  println!("{v:?}");
  for i in &mut v {
    *i += 100;
  }
  println!("After mutating iteration: {v:?}");
}

fn vec_variant_test() {
  #[derive(Debug)]
  enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
  }
  use SpreadsheetCell::Int as SInt;
  use SpreadsheetCell::Float as SFloat;
  use SpreadsheetCell::Text as SText;
  let row = vec![
    SFloat(2.0),
    SInt(3),
    SText(String::from("blue")),
  ];
  println!("Row: {row:?}");
}

fn string_create_test() {
  let s = "String value".to_string();
  println!("Str: {s}");
  let s = String::from("Hello World!");
  println!("Str: {s}");
  let s = String::from("Здравствуйте");
  println!("Str: {s}");
  let s = String::from("नमस्ते");
  println!("Str: {s}");
}

fn string_update_test() {
  let mut s = String::new();
  s.push_str("Hello");
  s.push(' ');
  let s2 = "Universe!";
  s.push_str(s2);
  println!("Str: {s} s2: {s2}");
}

fn string_concat_test() {
  let s1 = "Hello ".to_string();
  let s2 = "world!".to_string();
  let s3 = s1 + &s2;
  println!("s2: {s2} s3: {s3}");
  let s1 = "bang";
  let s2 = "maxwell";
  let s3 = format!("{s1}-{s1}-{}", s2);
  println!("s3: {s3}");
}

fn string_index_test() {
  // Now allowed to index directly into String
  // let s1 = String::from("hello");
  // let h = s[0];
  // println!("First char of {s1} is {h}");

  let s = String::from("Здравствуйте");
  println!("size of {s} is {}", s.len());
  // Causes panic
  // println!("First byte slice of {s} is {}", &s[0..1]);
  println!("Second char slice of {s} is {}", &s[2..4]);

  let str = "नमस्ते";
  for (i, c) in str.chars().enumerate() {
    println!("Char of {str} at {i}: {c}");
  }

  for (i, b) in str.bytes().enumerate() {
    println!("Byte of {str} at {i}: {b}");
  }
}

fn string_slices_test() {
  let s = String::from("hello world");
  let hello = &s[0..5];
  let hello1 = &s[..5];
  let world = &s[6..11];
  let world1 = &s[6..];

  println!("Slices of {s} are {hello} and {world}");
  println!("Slices of {s} are {hello1} and {world1}");
}

fn first_word(s: &str) -> &str {
  let bytes = s.as_bytes();
  for (i, &item) in bytes.iter().enumerate() {
    if item == b' ' {
      return &s[..i];
    }
  }
  &s[..]
}

fn use_slice() {
  let /*mut*/ s = String::from("Hello world");
  let word = first_word(&s);
  // s.clear();   // Error!
  println!("First word: {word}");

  first_word(&s[0..6]);
  first_word(&s[..]);
  first_word(&s);

  let lit_str = "hello world";
  first_word(&lit_str[0..6]);
  first_word(&lit_str[..]);
  first_word(lit_str);
}

use std::collections::HashMap;

fn hash_map_create_test() {
  let mut scores = HashMap::new();
  scores.insert(String::from("Thanos"), 5);
  scores.insert(String::from("Avengers"), 2);

  println!("Scores: {scores:?}");

  let team_scores: HashMap<String, i32> = vec![
    (String::from("Thanos"), 5),
    (String::from("Avengers"), 2),
  ].into_iter().collect();
  println!("Scores: {team_scores:?}");

  let mut sqr = HashMap::new();
  sqr.insert(1, 1);
  sqr.insert(2, 4);
  sqr.insert(3, 9);
  println!("sqr table: {sqr:?}");

  let field_name = String::from("field_name");
  let field_value = String::from("field_value");
  let mut map = HashMap::new();
  map.insert(field_name, field_value);
  println!("map: {map:?}");
  // Error since field_name has been moved
  // println!("field_name: {field_name}");
}

fn hash_map_access_test() {
  let mut team_scores: HashMap<String, i32> = vec![
    (String::from("Thanos"), 5),
    (String::from("Avengers"), 2),
  ].into_iter().collect();
  println!("Scores: {team_scores:?}");

  println!("score for Thanos: {:?}, score for fool: {:?}",
    team_scores.get("Thanos"),
    team_scores.get("fool"),
  );

  for (k, v) in &team_scores {
    println!("k={k} v={v}");
  }

  team_scores.insert("Avengers".to_string(), 6);
  println!("Scores: {team_scores:?}");

  team_scores.entry(String::from("Thanos")).or_insert(10);
  team_scores.entry(String::from("XMen")).or_insert(5);
  println!("Scores: {team_scores:?}");

  let th_score = team_scores.entry(String::from("Thanos")).or_insert(10);
  *th_score += 10;
  println!("Scores: {team_scores:?}");
}

fn main() {
  vec_create_test();
  vec_read_test();
  vec_iter_test();
  vec_variant_test();

  string_create_test();
  string_update_test();
  string_concat_test();
  string_index_test();
  string_slices_test();
  use_slice();

  hash_map_create_test();
  hash_map_access_test();
}
