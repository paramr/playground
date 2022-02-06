#![allow(dead_code)]

#[derive(Debug)]
struct Rectangle {
  width: u32,
  height: u32,
}

impl Rectangle {
  fn can_hold(
    &self,
    other: &Rectangle,
  ) -> bool {
    self.width > other.width && self.height > other.height
  }
}

fn i_panic() {
  panic!("i_panic message");
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn it_works() {
    let result = 2 + 2;
    assert_eq!(result, 4);
    assert_ne!(result, 3);
  }

  #[test]
  fn larger_can_hold_smaller() {
    let larger = Rectangle {
      width: 8,
      height: 7,
    };
    let smaller = Rectangle {
      width: 5,
      height: 1,
    };

    assert!(
      larger.can_hold(&smaller),
      "rectangle: {:?} does not hold {:?}",
      larger,
      smaller,
    );
  }

  #[test]
  #[should_panic(expected = "i_panic message")]
  fn test_i_panic() {
    i_panic();
  }
}
