#![allow(dead_code)]

#[derive(Debug)]
struct Point<T> {
  x: T,
  y: T,
}

impl<T> Point<T> {
  fn x(&self) -> &T {
    &self.x
  }
}

impl Point<f64> {
  fn distance_from_origin(&self) -> f64 {
    (self.x.powi(2) + self.y.powi(2)).sqrt()
  }
}

#[derive(Debug)]
struct Point2<X1, Y1> {
  x: X1,
  y: Y1,
}

impl<X1, Y1> Point2<X1, Y1> {
  fn mixup<X2, Y2>(self, other: Point2<X2, Y2>) -> Point2<X1, Y2> {
    Point2 {
      x: self.x,
      y: other.y,
    }
  }
}

fn test_generic_struct() {
  let int_point = Point { x: 5, y: 10 };
  let float_point = Point { x: 1.0, y: 4.0 };
  // let bad_point = Point { x: 10, y: 4.0 };
  println!("int_point: {int_point:?} float_point: {float_point:?}");
  println!("int_point.x: {:?}", int_point.x());
  println!("float_point dist: {:?}", float_point.distance_from_origin());

  let p1 = Point2 { x: 5, y: 10.5 };
  let p2 = Point2 { x: 'H', y: String::from("Hello") };
  let p3 = p1.mixup(p2);
  println!("p3: {p3:?}");
}

#[derive(Debug)]
enum Opt<T> {
  Some(T),
  None,
}

fn pack_opt<T>(t: T) -> Opt<T> {
  Opt::Some(t)
}

fn none_opt<T>() -> Opt<T> {
  Opt::None
}

fn test_genric_enum() {
  println!("pack_opt: {:?}", pack_opt(5));
  println!("none_opt: {:?}", none_opt::<u32>());
}

fn largest<T: PartialOrd>(list: &[T]) -> &T {
  let mut largest = &list[0];

  for item in list {
      if item > largest {
          largest = item;
      }
  }

  largest
}

fn test_largest() {
  let number_list = vec![34, 50, 25, 100, 65];
  let result = largest(&number_list);
  println!("The largest number is {}", result);

  let char_list = vec!['y', 'm', 'a', 'q'];
  let result = largest(&char_list);
  println!("The largest char is {}", result);
}

// default for generic param
trait Add<T=Self> {
  fn add(self, other: T) -> T;
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Point2D {
  x: f64,
  y: f64,
}

impl Add for Point2D {
  fn add(self, other: Point2D) -> Point2D {
    Point2D {
      x: self.x + other.x,
      y: self.y + other.y,
    }
  }
}

fn test_point2d_add() {
  let p = Point2D { x: 1.0, y: 2.0 };
  let p1 = p.add(Point2D { x: 1.0, y: 1.0 });
  println!("p: {p:?} p1: {p1:?}");
}

fn main() {
  test_generic_struct();
  test_genric_enum();
  test_largest();
  test_point2d_add();
}
