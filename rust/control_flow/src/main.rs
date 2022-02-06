#![allow(dead_code)]

fn if_test() {
  let number = 3;
  if number < 5 {
    println!("number < 5");
  } else if number > 5{
    println!("number > 5");
  } else {
    println!("number = 5");
  }

  let a_num = if number < 3 { 5 } else { 7 };
  println!("a_num = {a_num}");
}

fn loop_test1() {
  let mut count = 0;
  'label1: loop {
    println!("count: {count}");
    let mut remaining = 10;
    loop {
      println!("remaining: {}", remaining);
      if remaining == 9 {
        break;
      }
      if count == 2 {
        break 'label1;
      }
      remaining -= 1;
    }
    count += 1;
  }
  println!("end count: {count}");
}

fn loop_test2() {
  let mut counter = 0;
  let result = loop {
    counter += 1;
    if counter == 10 {
      break counter * 2;
    }
  };
  println!("The result is: {result}");
}

fn while_test1() {
  let mut num = 5;
  while num >= 0 {
    println!("{num}!");
    num -= 1;
  }
  println!("LIFTOFF!!!");
}

fn while_test2() {
  let arr = [10, 20, 30, 40, 50];
  let mut index = 0;
  while index < arr.len() {
    println!("The value at index {index} is: {}", arr[index]);
    index += 1;
  }
}

fn for_test1() {
  let arr = [10, 20, 30, 40, 50];
  for (index, element) in arr.iter().enumerate() {
    println!("The value is arr[{index}] is {element}");
  }
}

fn for_test2() {
  for num in (1..4).rev() {
    println!("{num}!");
  }
  println!("LIFTOFF!!!")
}

fn main() {
  if_test();
  loop_test1();
  loop_test2();
  while_test1();
  while_test2();
  for_test1();
  for_test2();
}
