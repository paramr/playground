#[macro_export]
macro_rules! myvec {
  ($( $x: expr),*) => {
    {
      let mut temp_vec = Vec::new();
      $(
        temp_vec.push($x);
      )*
      temp_vec
    }
  }
}

fn test_myvec() {
  let v = myvec!(1, 2, 3);
  println!("v: {v:?}");
}

fn main() {
  test_myvec();
}
