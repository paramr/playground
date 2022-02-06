#![allow(dead_code)]
#![allow(unused_imports)]

mod front_of_house;

mod data {
  pub struct Menu {
    pub item1: String,
    item2: String,
  }
}

fn eat_at_restaurant() {
  crate::front_of_house::hosting::add_to_waitlist();
  self::front_of_house::hosting::seat_at_table();
  front_of_house::hosting::take_menu();

  // Error no access to item2.
  // let menu = data::Menu {
  //   item1: String::from("Mango"),
  //   item2: String::from("Samosa"),
  // };
}

use self::front_of_house::pos_system;
// use self::front_of_house::serving;   // Error because of privacy.

fn use_test() {
  use crate::front_of_house::hosting;
  use crate::front_of_house::hosting::add_to_waitlist;

  add_to_waitlist();
  hosting::take_menu();
}

use std::fmt;
use std::io;

fn f1() -> fmt::Result {
  fmt::Result::Ok(())
}

fn f2() -> io::Result<()> {
  io::Result::Ok(())
}

pub use fmt::Result as FmtResult;
pub use io::Result as IoResult;

fn f3() -> FmtResult {
  f1()
}

fn f4() -> IoResult<()> {
  f2()
}
