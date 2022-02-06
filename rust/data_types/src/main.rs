#![allow(dead_code)]

fn integer_types() {
  let vnum_i32: i32 = 98_222;
  println!("Value of vnum_i32 is: {vnum_i32}");
  let vhex_u128: u128 = 0xFF;
  println!("Value of vhex_u128 is: {vhex_u128}");
  let voct_u128: u128 = 0o77;
  println!("Value of voct_u128 is: {voct_u128}");
  let vbin_u128 = 0b1111_0000;
  println!("Value of vbin_u128 is: {vbin_u128}");
  let v_byte = b'A';
  println!("Value of v_byte is: {v_byte}");
}

fn floating_types() {
  let x = 2.0;

  let y: f32 = 3.0;

  println!("Value of x: {x} an y: {y}");
}

fn numeric_operations() {
  let sum = 5 + 10;
  println!("sum is: {sum}");
  let diff = 44.4 - 33.2;
  println!("diff is: {diff}");
  let prod = 4 * 30;
  println!("prod is: {prod}");
  let quotient = 56.7 / 32.2;
  println!("quotient is: {quotient}");
  let floored = 2 / 3;
  println!("floored is: {floored}");
  let remainder = 43 % 5;
  println!("remainder is: {remainder}");
}

fn boolean_types() {
  let t = true;
  let f: bool = false;
  println!("t is: {t} and f is {f}");
}

fn char_types() {
  let c = 'z';
  let z = 'ℤ';
  let heart_eyed_cat = '😻';

  println!("c is {c}, x is {z} and heart_eyed_cat is {heart_eyed_cat}");
}

fn tuple_types() {
  let tup: (i32, f32, char) = (1, 2.2, 'Z');
  let (x, y, z) = tup;

  println!("tup is {:?}", tup);
  println!("tup is ({x}, {y}, {z})");
  println!("tup is also ({}, {}, {})", tup.0, tup.1, tup.2);
}

fn array_type() {
  let a = [1, 2, 3, 4, 5];
  let b: [i16; 5] = [1, 2, 3, 4, 5,];
  let c = [4; 5];
  println!("a is {:?}", a);
  println!("b[0] is {}", b[0]);
  println!("c is {:?}", c);
}


#[derive(Debug)]
enum IpAddrKind {
  V4,
  V6,
}

#[derive(Debug)]
struct IpAddrStruct {
  kind: IpAddrKind,
  address: String,
}

impl IpAddrStruct {
  fn from(kind: IpAddrKind, address: String) -> Self {
    Self { kind, address }
  }
}

fn print_ip_addr_struct(ip_addr: IpAddrStruct) {
  println!("IpAddr: {:?}", ip_addr);
}

#[derive(Debug)]
enum IpAddrStr {
  V4(String),
  V6(String),
}

fn print_ip_addr_str(ip_addr: IpAddrStr) {
  println!("IpAddr: {:?}", ip_addr);
}

#[derive(Debug)]
enum IpAddr {
  V4(u8, u8, u8, u8),
  V6(String),
}

fn print_ip_addr(ip_addr: IpAddr) {
  println!("IpAddr: {:?}", ip_addr);
}

enum Message {
  Quit,
  Move { x: i32, y: i32 },
  Write(String),
  ChangeColor(i32, i32, i32),
}

impl Message {
  fn make_quit() -> Message {
    Message::Quit
  }
}

fn option_test() {
  let op_none: Option<i32> = None;
  let op_5 = Some(5);
  println!("op_none: {op_none:?} op_5: {op_5:?}");
  let mut opt: Option<i32> = None;
  let val = opt.insert(1);
  *val = 3;
  println!("Opt: {opt:?}");
}

#[derive(Debug)]
#[derive(Clone)]
struct User {
  active: bool,
  username: String,
  email: String,
  sign_in_count: u64,
}

fn struct_create_test() {
  let user1 = User {
    email: String::from("hello@hell.com"),
    username: String::from("hello_hell"),
    active: true,
    sign_in_count: 1,
  };

  println!("User1 is: {:?}", user1);
  println!("Email for user1 is: {}", user1.email);

  let mut user2 = build_user(user1.email, user1.username);
  println!("User2 is: {:?}", user2);
  user2.email = String::from("hell@hello.com");
  println!("New User2 is: {:?}", user2);
}

fn build_user(email: String, username: String) -> User {
  User {
    email,  //  same as email: email
    username, //  same as username: username
    active: true,
    sign_in_count: 1,
  }
}

fn struct_update_test() {
  let user1 = build_user(
    String::from("hello@hell.com"),
    String::from("hello_hell"),
  );
  let user1_clone = user1.clone();

  // Update syntax moves from user1 into user2 in this case
  let user2 = User {
    email: String::from("new_hello@hell.com"),
    ..user1
  };
  // println!("User1 is: {:?}", user1); //  Error
  println!("User2 is: {user2:?}");

  // Update syntax does not move from user1_clone into user3
  // in this case
  let user3 = User {
    email: dbg!(String::from("new_hello@hell.com")),
    username: String::from("new_hello_hell"),
    ..user1_clone
  };
  println!("User3 is: {user3:?}");
  dbg!(user2, user3);
}

#[derive(Debug)]
struct Color(i32, i32, i32);

#[derive(Debug)]
struct Point(i32, i32);

#[derive(Debug)]
struct UnitStruct;

fn tuple_struct_test() {
  let black = Color(0, 0, 0);
  let origin = Point(0, 0);
  let unit = UnitStruct;
  println!("Black is: {black:?}");
  println!("Origin is: {origin:?}");
  println!("Unit is: {unit:?}");
}

#[derive(Debug)]
struct Rectangle {
  width: u32,
  height: u32,
}

impl Rectangle {
  #[allow(dead_code)]
  fn from(width: u32, height: u32) -> Self {
    Self {
      width,
      height,
    }
  }

  #[allow(dead_code)]
  fn square(size: u32) -> Rectangle {
    Rectangle {
      width: size,
      height: size,
    }
  }

  fn area(&self) -> u32 {
    self.width * self.height
  }

  fn width(&self) -> bool {
    self.width > 0
  }

  fn can_hold(&self, other: &Rectangle) -> bool {
    self.width > other.width && self.height > other.height
  }
}

impl Rectangle {
  fn perimeter(&self) -> u32 {
    2 * (self.width + self.height)
  }

  fn scale(&mut self, factor: f64) {
    self.height = (self.height as f64 * factor).round() as u32;
    self.width = (self.width as f64 * factor).round() as u32;
  }
}

fn struct_method_test() {
  let mut rect = Rectangle {
    width: 30,
    height: 50,
  };
  println!(
    "The rect {rect:?} has area {} and perimeter {}.",
    rect.area(),
    Rectangle::perimeter(&rect),
  );
  rect.scale(0.5);
  let rect2 = Rectangle {
    width: 10,
    height: 10,
  };
  println!(
    "New Rect: {rect:?}, has width {} and contains {rect2:?} {}",
    rect.width(),
    rect.can_hold(&rect2),
  );
}

trait Summary {
  // By default use short_summary
  fn summary(&self) -> String {
    self.short_summary()
  }

  fn short_summary(&self) -> String;
}

#[derive(Debug)]
struct Topic {
  pub name: String,
  pub author: String,
  pub desc: String,
}

#[derive(Debug)]
struct Theorem {
  pub name: String,
  pub author: String,
  pub statement: String,
  pub proof: String,
}

impl Summary for Topic {
  fn short_summary(&self) -> String {
    format!("{} - {}", self.name, self.author)
  }
}

impl Summary for Theorem {
  fn short_summary(&self) -> String {
    format!("{} - {}", self.name, self.author)
  }

  fn summary(&self) -> String {
    format!("{} - {} - {}", self.name, self.author, self.statement)
  }
}

fn present_result_short_summary(item: &impl Summary) {
  println!("short_summary: {}", item.short_summary());
}

fn present_result_summary<T: Summary>(item: &T) {
  println!("summary: {}", item.summary());
}

use std::fmt::Debug;

fn present_debug_and_short_summary_v1(item: &(impl Summary + Debug)) {
  present_debug_and_short_summary_v2(item);
}

fn present_debug_and_short_summary_v2<T: Summary + Debug>(item: &T) {
  present_debug_and_short_summary_v3(item);
}

fn present_debug_and_short_summary_v3<T>(item: &T)
  where T: Summary + Debug {
  println!("Dbg: {item:?} short_summary: {}", item.short_summary());
}

fn test_trait() {
  let topic = Topic {
    name: String::from("geometry"),
    author: String::from("Euclid"),
    desc: String::from("Thirteen elements"),
  };
  present_debug_and_short_summary_v1(&topic);
  present_result_short_summary(&topic);
  present_result_summary(&topic);
  println!("For {topic:?}");

  let theorem = Theorem {
    name: String::from("Pythagoras theorem"),
    author: String::from("Pythagoras"),
    statement: String::from("h^2 = a^2 + b^2"),
    proof: String::from("By authority"),
  };
  present_debug_and_short_summary_v1(&theorem);
  present_result_short_summary(&theorem);
  present_result_summary(&theorem);
  println!("For {theorem:?}");
}

pub trait Draw {
  fn draw(&self);
}

pub struct Screen {
  pub components: Vec<Box<dyn Draw>>,
}

pub struct Button(String);

impl Draw for Button {
  fn draw(&self) {
    println!("draw button: {}", self.0);
  }
}

pub struct SelectBox(String);

impl Draw for SelectBox {
  fn draw(&self) {
    println!("draw selectbox: {}", self.0);
  }
}

pub struct DuckBox(String);

impl DuckBox {
  #[allow(dead_code)]
  fn draw(&self) {
    println!("draw duckbox: {}", self.0);
  }
}

impl Screen {
  pub fn draw(&self) {
    for component in self.components.iter() {
      component.draw();
    }
  }
}

fn test_dyn_traits() {
  let screen = Screen {
    components: vec![
      Box::new(SelectBox(String::from("select_name_1"))),
      Box::new(Button(String::from("button_1"))),
      Box::new(Button(String::from("button_2"))),
      Box::new(SelectBox(String::from("select_name_2"))),
      // Not really Ducktyping
      // Box::new(DuckBox(String::from("select_name_2"))),
    ],
  };
  screen.draw();
}

trait State {
  fn request_review(self: Box<Self>) -> Box<dyn State>;
  fn approve(self: Box<Self>) -> Box<dyn State>;
  fn content<'a>(&self, _post: &'a Post) -> &'a str {
    ""
  }
}

struct Draft {}

impl State for Draft {
  fn request_review(self: Box<Self>) -> Box<dyn State> {
    Box::new(PendingReview {})
  }

  fn approve(self: Box<Self>) -> Box<dyn State> {
    self
  }
}

struct PendingReview {}

impl State for PendingReview {
  fn request_review(self: Box<Self>) -> Box<dyn State> {
    self
  }

  fn approve(self: Box<Self>) -> Box<dyn State> {
    Box::new(Published {})
  }
}

struct Published {}

impl State for Published {
  fn request_review(self: Box<Self>) -> Box<dyn State> {
    self
  }

  fn approve(self: Box<Self>) -> Box<dyn State> {
    self
  }

  fn content<'a>(&self, post: &'a Post) -> &'a str {
    &post.content
  }
}

struct Post {
  state: Option<Box<dyn State>>,
  content: String,
}

impl Post {
  pub fn new() -> Post {
    Post {
      state: Some(Box::new(Draft {})),
      content: String::new(),
    }
  }

  pub fn add_text(&mut self, text: &str) {
    self.content.push_str(text);
  }

  pub fn request_review(&mut self) {
    if let Some(s) = self.state.take() {
      self.state = Some(s.request_review());
    }
  }

  pub fn approve(&mut self) {
    if let Some(s) = self.state.take() {
      self.state = Some(s.approve());
    }
  }

  pub fn content(&self) -> &str {
    self.state.as_ref().unwrap().content(self)
  }
}

fn test_post_state() {
  let mut post = Post::new();

  post.add_text("Hello post");
  assert_eq!(post.content(), "");

  post.request_review();
  assert_eq!(post.content(), "");

  post.approve();
  assert_eq!(post.content(), "Hello post");
}

pub struct RPost {
  content: String,
}

impl RPost {
  pub fn new() -> RDraftPost {
    RDraftPost {
      content: String::new(),
    }
  }

  pub fn content(&self) -> &str {
    &self.content
  }
}

pub struct RDraftPost {
  content: String,
}

impl RDraftPost {
  pub fn add_text(&mut self, text: &str) {
    self.content.push_str(text);
  }

  pub fn request_review(self) -> RPendingReviewPost {
    RPendingReviewPost {
      content: self.content,
    }
  }
}

pub struct RPendingReviewPost {
  content: String,
}

impl RPendingReviewPost {
  pub fn approve(self) -> RPost {
    RPost {
      content: self.content,
    }
  }
}

fn test_post_state_types() {
  let mut post = RPost::new();
  post.add_text("Hello universe!");
  let post = post.request_review();
  let post = post.approve();
  assert_eq!(post.content(), "Hello universe!");
}

trait SingleImplTrait {
  type ItemType;

  fn vect(&self) -> Vec<Self::ItemType>;
}

trait MultiImplTrait<T> {
  fn vect(&self) -> Vec<T>;
}

struct TraitHolder;

impl TraitHolder {
  fn vect(&self) -> Vec<i32> {
    vec![10, 20, 30]
  }
}

impl SingleImplTrait for TraitHolder {
  type ItemType = String;

  fn vect(&self) -> Vec<Self::ItemType> {
    vec![
      String::from("Hello"),
      String::from("World"),
    ]
  }
}

impl MultiImplTrait<String> for TraitHolder {
  fn vect(&self) -> Vec<String> {
    vec![
      String::from("elloH"),
      String::from("orldW"),
    ]
  }
}

impl MultiImplTrait<i32> for TraitHolder {
  fn vect(&self) -> Vec<i32> {
    vec![1, 2, 3]
  }
}

fn test_assoc_traits() {
  let th = TraitHolder {};

  for i in th.vect() {
    println!("th i is {i}");
  }
  for i in TraitHolder::vect(&th) {
    println!("TraitHolder i is {i}");
  }

  let sit = &th as &dyn SingleImplTrait<ItemType = String>;
  for s in sit.vect() {
    println!("sit s is {s}");
  }
  for s in SingleImplTrait::vect(&th) {
    println!("SingleImplTrait s is {s}");
  }

  let mit_str = &th as &dyn MultiImplTrait<String>;
  for s in mit_str.vect() {
    println!("mit_str s is {s}");
  }
  for s in MultiImplTrait::<String>::vect(&th) {
    println!("MultiImplTrait<String> s is {s}");
  }

  let mit_i32 = &th as &dyn MultiImplTrait<i32>;
  for i in mit_i32.vect() {
    println!("mit_i32 i is {i}");
  }
  for i in MultiImplTrait::<i32>::vect(&th) {
    println!("MultiImplTrait<i32> i is {i}");
  }
}

trait Animal {
  fn name() -> String;
}

struct Dog;

impl Dog {
  fn name() -> String {
    String::from("Spot")
  }
}

impl Animal for Dog {
  fn name() -> String {
    String::from("puppy")
  }
}

fn test_ambiguous_trait_func_call() {
  println!("Dog name is: {}", Dog::name());
  // Does not work
  // println!("A baby dog is: {}", Animal::name());
  println!("Dog name is: {}", <Dog as Animal>::name());
}

// Dependent traits

trait OutlinePrint: std::fmt::Display {
  fn outline_print(&self) {
    let output = self.to_string();
    let len = output.len();
    println!("{}", "*".repeat(len + 4));
    println!("*{}*", " ".repeat(len + 2));
    println!("* {} *", output);
    println!("*{}*", " ".repeat(len + 2));
    println!("{}", "*".repeat(len + 4));
  }
}

impl std::fmt::Display for Point {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
      write!(f, "({}, {})", self.0, self.1)
  }
}

impl OutlinePrint for Point {}

fn test_dependent_type() {
  let p = Point(1, 2);
  p.outline_print();
}

struct Wrapper(Vec<String>);

impl std::fmt::Display for Wrapper {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "[{}]", self.0.join(", "))
  }
}

// Implement deref to expose Vec methods on Wrapper
impl std::ops::Deref for Wrapper {
  type Target = Vec<String>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

fn test_new_type_pattern() {
  let w = Wrapper(vec![
    String::from("hello"),
    String::from("universe"),
  ]);
  println!("w: {w} with length: {} w[0]: {}", w.len(), w[0]);
}

type MagicInt = i32;

fn main() {
  integer_types();
  floating_types();
  numeric_operations();
  boolean_types();
  char_types();
  tuple_types();
  array_type();
  print_ip_addr_struct(IpAddrStruct::from(
    IpAddrKind::V4,
    String::from("127.0.0.0"),
  ));
  print_ip_addr_str(IpAddrStr::V6(String::from("::1")));
  print_ip_addr(IpAddr::V4(127, 0, 0, 0));
  option_test();
  struct_create_test();
  struct_update_test();
  tuple_struct_test();
  struct_method_test();
  test_trait();
  test_dyn_traits();
  test_post_state();
  test_post_state_types();
  test_assoc_traits();
  test_ambiguous_trait_func_call();
  test_dependent_type();
  test_new_type_pattern();
}
