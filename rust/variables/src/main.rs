const THREE_HOURS_IN_SECONDS: u32 = 3 * 60 * 60;

fn variables() {
  let mut x = 5;
  println!("THe value of x is: {x}");
  x = 6;
  println!("THe value of x is: {x}");
  println!("THe value of THREE_HOURS_IN_SECONDS is: {THREE_HOURS_IN_SECONDS}");
}

fn shadowing() {
  let x = 5;

  let x = x + 1;

  {
    let x = x * 2;
    println!("The value of x in inner scope is: {x}");
  }

  println!("The value of x is: {x}");
}

fn references() {
  let mut x = 5;
  println!("x before: {x}");
  let xr = &mut x;
  *xr = 1;
  println!("x after: {x}");
}

fn main() {
  variables();
  shadowing();
  references();
}