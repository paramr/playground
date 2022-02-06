#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

fn countdown(
  x: i32,
  limit: i32,
) {
  if x < limit {
    return;
  }
  println!("countdown: {x}");
  countdown(x - 1, limit);
}

fn expr_block() {
  let y = {
    let x = 3;
    x + 1
  };

  println!("The value of y is: {y}");
}

fn fact(n: i32) -> i32 {
  return if n == 0 { 1 } else { n * fact(n - 1) };
}

// use std::thread;
// use std::time::Duration;

struct Cacher<T>
where
  T: Fn(u32) -> u32,
{
  calc: T,
  value: Option<u32>,
}

impl<T> Cacher<T>
where
  T: Fn(u32) -> u32,
{
  fn new(calc: T) -> Cacher<T> {
    Cacher { calc, value: None }
  }

  fn value(
    &mut self,
    arg: u32,
  ) -> u32 {
    match self.value {
      Some(v) => v,
      None => {
        let v = (self.calc)(arg);
        self.value = Some(v);
        v
      }
    }
  }
}

fn generate_workout(
  intensity: u32,
  random_number: u32,
) {
  let mut cached_closure = Cacher::new(|num| {
    println!("calculating slowly...");
    // thread::sleep(Duration::from_secs(2));
    num
  });
  if intensity < 25 {
    println!("Today, do {} pushups!", cached_closure.value(intensity));
    println!("Next, do {} situps!", cached_closure.value(intensity));
  } else {
    if random_number == 3 {
      println!("Take a break today! Remember to stay hydrated!");
    } else {
      println!(
        "Today, run for {} minutes!",
        cached_closure.value(intensity)
      );
    }
  }
}

fn test_closure() {
  let intensity = 10;
  let rand = 7;

  generate_workout(intensity, rand);
}

fn test_capture() {
  let x = 4;
  let equal_to_x = |z| z == x;
  assert!(equal_to_x(4));
  println!("x is {x}");

  let x = vec![1, 2, 3];
  let equal_to_x = |z: Vec<i32>| z == x;
  println!("x is {:?}", x);

  let mut x = 5;
  let equal_to_x = move |z| z == x;
  assert!(equal_to_x(5));
  x = 4;
  assert!(equal_to_x(5));

  let x = vec![1, 2, 3];
  let equal_to_x = move |z: Vec<i32>| z == x;
  // println!("x is {:?}", x);  //  Error here
}

fn test_iterator_int() {
  let v1 = vec![1, 2, 3];
  let v1_iter = v1.iter();
  for v in v1_iter {
    println!("Got: {v}");
  }

  let mut v1_iter = v1.iter();
  assert_eq!(v1_iter.next(), Some(&1));
  assert_eq!(v1_iter.next(), Some(&2));
  assert_eq!(v1_iter.next(), Some(&3));
  assert_eq!(v1_iter.next(), None);

  let v1_iter = v1.iter();
  let total: i32 = v1_iter.sum();
  assert_eq!(total, 6);

  let vt: Vec<_> = v1.iter().map(|x| 2 * x).collect();
  assert_eq!(vt, vec![2, 4, 6]);

  let vt: Vec<_> = v1.iter().filter(|n| *n % 2 == 1).map(|n| *n).collect();
  assert_eq!(vt, vec![1, 3]);
  assert_eq!(v1, vec![1, 2, 3]);

  let vt: Vec<_> = v1.into_iter().filter(|n| *n % 2 == 1).collect();
  assert_eq!(vt, vec![1, 3]);
  // assert_eq!(v1, vec![1, 2, 3]);
}

fn test_iterator_str() {
  let v1 = vec![String::from("a"), String::from("bc"), String::from("def")];
  let v1_iter = v1.iter();
  for v in v1_iter {
    println!("Got: {v}");
  }

  let mut v1_iter = v1.iter();
  assert_eq!(v1_iter.next(), Some(&String::from("a")));
  assert_eq!(v1_iter.next(), Some(&String::from("bc")));
  assert_eq!(v1_iter.next(), Some(&String::from("def")));
  assert_eq!(v1_iter.next(), None);

  let vt: Vec<_> = v1.iter().map(|x| x.clone() + "1").collect();
  println!("vt: {vt:?}");

  let vt: Vec<_> = v1
    .iter()
    .filter(|n| n.len() % 2 == 1)
    .map(|n| n.clone())
    .collect();
  println!("vt: {vt:?} v1: {v1:?}");
  println!("v1: {v1:?}");

  let vt: Vec<_> = v1.into_iter().filter(|n| n.len() % 2 == 1).collect();
  println!("vt: {vt:?}");
  // println!("v1: {v1:?}");  //  Error
}

struct Counter {
  from: u32,
  to: u32,
}

impl Counter {
  fn new(
    from: u32,
    to: u32,
  ) -> Counter {
    Counter { from, to }
  }
}

impl Iterator for Counter {
  type Item = u32;

  fn next(&mut self) -> Option<Self::Item> {
    if self.from < self.to {
      let ret = self.from;
      self.from += 1;
      Some(ret)
    } else {
      None
    }
  }
}

fn test_counter() {
  let counter = Counter::new(0, 3);
  for i in counter {
    println!("i is {i}");
  }

  let mut counter = Counter::new(0, 3);
  assert_eq!(counter.next(), Some(0));
  assert_eq!(counter.next(), Some(1));
  assert_eq!(counter.next(), Some(2));
  assert_eq!(counter.next(), None);
  assert_eq!(counter.next(), None);

  let nv: Vec<_> = Counter::new(1, 4)
    .zip(Counter::new(10, 13))
    .map(|(a, b)| (a, b, a + b, a * b))
    .collect();
  println!("Zipped: {nv:?}");
}

fn add_once(x: i32) -> i32 {
  x + x
}

fn mul_once(x: i32) -> i32 {
  x * x
}

fn repeated_caller_fuc(
  f: fn(i32) -> i32,
  rep: i32,
  arg: i32,
) -> i32 {
  let mut ret = arg;
  for i in 0..rep {
    ret = f(ret);
  }
  ret
}

fn test_func_pointer() {
  println!(
    "repeated_caller(add_once, 3, 2): {}",
    repeated_caller_fuc(add_once, 3, 2),
  );
  println!(
    "repeated_caller(mul_once, 3, 2): {}",
    repeated_caller_fuc(mul_once, 3, 2),
  );
}

fn repeated_caller_closure(
  f: &dyn Fn(i32) -> i32,
  rep: i32,
  arg: i32,
) -> i32 {
  let mut ret = arg;
  for i in 0..rep {
    ret = f(ret);
  }
  ret
}

fn test_func_closure() {
  println!(
    "repeated_caller(add_once, 3, 2): {}",
    repeated_caller_closure(&add_once, 3, 2),
  );
  let add_once_closure = |n| n + n;
  println!(
    "repeated_caller(add_once_closure, 3, 2): {}",
    repeated_caller_closure(&add_once_closure, 3, 2),
  );
}

#[derive(Debug)]
struct Holder<T>(T);

fn test_enum_struct_tuple_init() {
  let hl: Vec<Holder<i32>> = (0..3).map(Holder).collect();
  let rl: Vec<Result<i32, ()>> = (0..3).map(Result::<i32, ()>::Ok).collect();
  println!("hl: {hl:?}");
  println!("rl: {rl:?}");
}

fn return_addone_closure() -> Box<dyn Fn(i32) -> i32> {
  Box::new(|x| x + 1)
}

fn test_return_addone_closure() {
  let addone_closure = return_addone_closure();
  println!("addone_closure(2): {}", addone_closure(2));
}

fn main() {
  countdown(5, 0);
  expr_block();
  println!("Factorial is: {}", fact(3));
  test_closure();
  test_capture();
  test_iterator_int();
  test_counter();
  test_func_pointer();
  test_func_closure();
  test_enum_struct_tuple_init();
  test_return_addone_closure();
}
