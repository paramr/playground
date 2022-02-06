pub fn prepare_serving() { }

fn take_order() {
  super::open_front_of_house();   // Access allowed
  super::super::front_of_house::hosting::take_menu();
}

fn serve_order() {}

fn take_payment() {}

mod internal {
  fn test_serving_system() {
    crate::front_of_house::open_front_of_house();
    super::super::open_front_of_house();
    super::serve_order();
  }
}
