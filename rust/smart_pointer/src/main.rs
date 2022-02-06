#![allow(dead_code)]
#![allow(unused_variables)]

#[derive(Debug)]
struct CityZip {
  city: String,
  zip: u32,
}

fn test_box() {
  let b = Box::new(5);
  println!("b = {}", b);
  println!("b = {}", &b);

  let cz = Box::<CityZip>::new(CityZip {
    city: String::from("Redmond"),
    zip: 98052,
  });
  println!("cityZip: {:?}", cz);
  println!("cityZip: {:?}", &cz);
}

#[derive(Debug)]
enum BList {
  Cons(i32, Box<BList>),
  Nil,
}

fn test_box_list() {
  let list = Box::new(BList::Cons(
    1,
    Box::new(BList::Cons(
      2,
      Box::new(BList::Cons(
        3,
        Box::new(BList::Nil),
      )),
    )),
  ));
  println!("list: {list:?}");
}

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

use std::ops::Deref;

impl<T> Deref for MyBox<T> {
  type Target = T;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

fn int_ref(x: &i32) {
  println!("x is {x}");
}

fn test_deref() {
  let x = 5;
  let y = MyBox::new(x);
  assert_eq!(5, x);
  assert_eq!(5, *(y.deref()));
  assert_eq!(5, *y);
  int_ref(&y);

  let box_box = MyBox::new(MyBox::new(2));
  int_ref(&box_box);
}

#[derive(Debug)]
struct MoveStruct(String);

impl Drop for MoveStruct {
  fn drop(&mut self) {
    println!("MoveStruct: {}", self.0);
  }
}

fn move_func(m: MoveStruct) {
  println!("move_func: {}", m.0);
}

fn box_func(bm: &mut Box<MoveStruct>) {
  println!("box_func: {}", bm.0);
  let ms = std::mem::replace(bm.as_mut(), MoveStruct(String::from("P6New")));
  println!("Exit: box_func");
}

fn test_drop_move() {
  let p1 = MoveStruct(String::from("P1"));
  let p1 = MoveStruct(String::from("P1New"));
  let p2 = MoveStruct(String::from("P2"));
  let p3 = MoveStruct(String::from("P3"));
  {
    let p4 = MoveStruct(String::from("P4"));
  }
  move_func(p3);
  // println!("p3: {p3:?}");
  drop(p2);
  // println!("p2: {p2:?}");
  let p5 = MoveStruct(String::from("P5"));
  let mut bp6 = Box::new(MoveStruct(String::from("P6")));
  box_func(&mut bp6);
}

use std::rc::{Rc, Weak};

fn test_rc() {
  {
    let rc = Rc::new(1);
    println!(
      "src/wrc count before clone: {}/{}",
      Rc::strong_count(&rc),
      Rc::weak_count(&rc),
    );
    let rc_clone = rc.clone();
    println!(
      "src/wrc count after clone: {}/{}",
      Rc::strong_count(&rc),
      Rc::weak_count(&rc),
    );
  }
  {
    let wrc: Weak<u32>;
    {
      let rc = Rc::new(1);
      println!(
        "src/wrc count before clone: {}/{}",
        Rc::strong_count(&rc),
        Rc::weak_count(&rc),
      );
      wrc = Rc::downgrade(&rc);
      println!(
        "src/wrc count after downgrade: {}/{}",
        Rc::strong_count(&rc),
        Rc::weak_count(&rc),
      );
      match wrc.upgrade() {
        Some(rc1) => {
          println!(
            "In Scope Some path src/wrc count after upgrade: {}/{}",
            Rc::strong_count(&rc),
            Rc::weak_count(&rc),
          );
        }
        None => {
          println!(
            "In Scope None path src/wrc count after upgrade: {}/{}",
            Rc::strong_count(&rc),
            Rc::weak_count(&rc),
          );
        }
      }
    }
    match wrc.upgrade() {
      Some(rc1) => {
        println!(
          "Out Scope Some path src/wrc count after upgrade: {}/{}",
          Rc::strong_count(&rc1),
          Rc::weak_count(&rc1),
        );
      }
      None => {
        println!("Out Scope None path src/wrc count after upgrade");
      }
    }
  }
}

// Error: destruct can not Copy
// #[derive(Debug, Copy, Clone)]
// struct CopyStruct(u32);

enum RcList {
  Cons(i32, Rc<RcList>),
  Nil,
}

fn test_rc_list() {
  let a = Rc::new(RcList::Cons(
    5,
    Rc::new(RcList::Cons(
      10,
      Rc::new(RcList::Nil)
    ))
  ));
  println!("After a count(a) = {}", Rc::strong_count(&a));
  let b = Rc::new(RcList::Cons(
    3,
    Rc::clone(&a)
  ));
  println!("After b count(a) = {}", Rc::strong_count(&a));
  {
    let c = Rc::new(RcList::Cons(
      4,
      Rc::clone(&a)
    ));
    println!("After c count(a) = {}", Rc::strong_count(&a));
  }
  println!("After scope out c count(a) = {}", Rc::strong_count(&a));
}

use std::cell::RefCell;

fn test_refcell() {
  let refcell_list = RefCell::new(vec![1, 2, 3]);
  // mutation is allowed
  refcell_list.borrow_mut().push(2);
  println!("refcell_list: {:?}", refcell_list.borrow());
  // Causes panic at runtime - 2 outstanding mutables
  // let bm = refcell_list.borrow_mut();
  // let bm = refcell_list.borrow_mut();

  // Causes panic at runtime - immutable and mutable outstanding
  // let bm = refcell_list.borrow_mut();
  // let b = refcell_list.borrow();

  // This is okay
  let b1 = refcell_list.borrow();
  let b2 = refcell_list.borrow();
}

#[derive(Debug)]
enum RefCellList {
    Cons(RefCell<i32>, Rc<RefCellList>),
    Nil,
}

fn test_refcell_list() {
  let a = Rc::new(RefCellList::Cons(
    RefCell::new(5),
    Rc::new(RefCellList::Nil),
  ));
  let b = Rc::new(RefCellList::Cons(
    RefCell::new(3),
    Rc::clone(&a),
  ));
  let c = Rc::new(RefCellList::Cons(
    RefCell::new(4),
    Rc::clone(&a),
  ));

  println!("a before = {a:?}");
  println!("b before = {b:?}");
  println!("c before = {c:?}");

  if let RefCellList::Cons(val, next) = a.as_ref() {
    *val.borrow_mut() += 10;
  }

  println!("a after = {a:?}");
  println!("b after = {b:?}");
  println!("c after = {c:?}");
}

fn main() {
  test_box();
  test_box_list();
  test_deref();
  test_drop_move();
  test_rc();
  test_rc_list();
  test_refcell();
  test_refcell_list();
}
