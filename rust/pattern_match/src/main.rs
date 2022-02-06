
#[derive(Debug)]
#[allow(dead_code)]
enum UsState {
  Alabama,
  Alaska,
  Washington,
  WestViginia,
  Wisconsin,
  Wyoming,
}

#[derive(Debug)]
#[allow(dead_code)]
enum Coin {
  Penny,
  Nickel,
  Dime,
  Quarter(UsState),
}

#[derive(Debug)]
struct FooF {
  x: i32,
  y: i32,
  z: String,
}

#[derive(Debug)]
struct FooT(i32, i32, String);

fn value_in_cents(coin: &Coin) -> u8 {
  match coin {
    Coin::Penny => {
      println!("Penny Lane!");
      1
    }
    Coin::Nickel => 5,
    Coin::Dime => 10,
    Coin::Quarter(state) => {
      println!("State: {state:?}");
      25
    }
  }
}

fn match_test() {
  let coin = Coin::Nickel;
  println!("Value of {coin:?} is {}", value_in_cents(&coin));
  let coin = Coin::Quarter(UsState::Washington);
  println!("Value of {coin:?} is {}", value_in_cents(&coin));
}

fn literal_match_test() {
  let p = FooF { x: 0, y: 7, z: String::from("Hello") };

  match p {
    FooF { x, y: 0, z } => {
      println!("1 x: {x}, y: 0, z: {z}");
    }
    FooF { x: 0, y, z } => {
      println!("2 x: 0, y: {y}, z: {z}");
    }
    FooF { x, z, .. } => {
      println!("3 x: {x}, z: {z}");
    }
  }
}

#[derive(Debug, Clone)]
struct EmbedInt {
  value: i32,
}

fn plus_one_max_5(x: Option<EmbedInt>) -> Option<EmbedInt> {
  match x {
    None => None,
    Some(ei) if ei.value < 5 => Some(EmbedInt {
      value: ei.value + 1,
    }),
    Some(_) => Some(EmbedInt{
      value: 5,
    }),
  }
}

fn option_match_test() {
  let four = Some(EmbedInt {
    value: 4,
  });
  let five = plus_one_max_5(four.clone());
  let other_five = plus_one_max_5(five.clone());
  let none = plus_one_max_5(None);

  println!("four: {four:?} five: {five:?} other_five: {other_five:?} None: {none:?}");
}

fn match_with_default_test() {
  let num = 9;
  match num {
    0 => println!("Zero!"),
    1 => println!("One!"),
    other => println!("Got: {other}"),
  }
  match num {
    0 => println!("Zero!"),
    1 => println!("One!"),
    _ => println!("Got something else"),
  }
  match num {
    -5..=0 => println!("less than eq Zero!"),
    1 | 2 => println!("One or two!"),
    _ => (),  // Do nothing
  }
}

fn if_let_test() {
  let opt_num = Some(5);
  if let Some(val) = opt_num {
    println!("Found Some(val) where val is {val}");
  }
}

fn while_let_test() {
  let mut stack = vec![1, 2, 3];

  while let Some(top) = stack.pop() {
    println!("{top}");
  }
}

fn for_test() {
  let v = vec!['a', 'b', 'c'];
  for (index, value) in v.iter().enumerate() {
    println!("{value} at {index}");
  }
}

fn let_test() {
  let tuple = (1, 2, String::from("Hell"));
  let (a, b, c) = tuple;
  println!("a: {a}, b: {b}, c: {c}");

  let foof = FooF { x: 1, y: 2, z: String::from("Hell") };
  let FooF { x: a, y: b, z: c } = foof;
  println!("a: {a}, b: {b}, c: {c}");

  let foof = FooF { x: 1, y: 2, z: String::from("Hell") };
  let FooF { x, y, z } = foof;
  println!("x: {x}, y: {y}, z: {z}");

  let foot = FooT(1, 2, String::from("Hell"));
  let FooT(a, b, c) = foot;
  println!("a: {a}, b: {b}, c: {c}");
}

fn func_pattern_ref(&(x, y): &(i32, i32)) {
  println!("func_pattern_ref ({x}, {y})");
}

fn func_pattern_struct(FooF { x: a, y: b, z: c }: FooF) {
  println!("func_pattern_struct a: {a}, b: {b}, c: {c}");
}

fn func_pattern_struct_sn(FooF { x, y, z }: FooF) {
  println!("func_pattern_struct x: {x}, y: {y}, z: {z}");
}

fn func_pattern_struct_ref(FooF { x: a, y: b, z: c }: &FooF) {
  println!("func_pattern_struct_ref a: {a}, b: {b}, c: {c}");
}

fn func_pattern_test() {
  let point = (3, 5);
  func_pattern_ref(&point);

  let foof = FooF { x: 1, y: 2, z: String::from("Hell") };
  func_pattern_struct_ref(&foof);
  func_pattern_struct(foof);

  let foof = FooF { x: 1, y: 2, z: String::from("Hell") };
  func_pattern_struct_sn(foof);
}

fn main() {
  match_test();
  literal_match_test();
  option_match_test();
  match_with_default_test();
  if_let_test();
  while_let_test();
  for_test();
  let_test();
  func_pattern_test();
}
