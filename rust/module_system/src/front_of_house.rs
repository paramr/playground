pub mod hosting;
pub mod serving;

fn open_front_of_house() {
  self::hosting::prepare_waitlist();
  self::serving::prepare_serving();
}

pub mod pos_system {
  fn test_pos() {}
}
