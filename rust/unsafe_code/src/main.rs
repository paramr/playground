#[allow(unused_variables)]

fn test_raw_pointer() {
  let mut num = 5;
  let p1 = &num as *const i32;
  let p2 = &mut num as *mut i32;
  unsafe {
    println!("value at p1 before: {}", *p1);
    *p2 = 10;
    println!("value at p1 after: {}", *p1);
  }

  let addr: usize = 0x01234567;
  let ap = addr as *mut i32;
  // Crashes
  // unsafe {
  //   println!("value at ap is: {}", *ap);
  //   *ap = 5;
  // }
}

unsafe fn unsafe_func() {
  let mut num = 5;
  let p1 = &num as *const i32;
  let p2 = &mut num as *mut i32;
  println!("unsafe_func value at p1 before: {}", *p1);
  *p2 = 10;
  println!("unsafe_func value at p1 after: {}", *p1);
}

fn test_unsafe_function() {
  unsafe {
    unsafe_func();
  }
}

fn unsafe_with_safe_wrapper(
  slice: &mut [i32],
  mid: usize,
) -> (&mut [i32], &mut [i32]) {
  let len = slice.len();
  let ptr = slice.as_mut_ptr();
  assert!(mid <= len);
  unsafe {
    (
      std::slice::from_raw_parts_mut(ptr, mid),
      std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
    )
  }
}

fn test_unsafe_with_safe_wrapper() {
  let mut v = vec![1, 2, 3, 4, 5, 6];
  {
    let (sl1, sl2) = unsafe_with_safe_wrapper(&mut v, 3);
    sl1[0] = 0;
    sl2[0] = 0;
    assert_eq!(sl1, &mut [0, 2, 3]);
    assert_eq!(sl2, &mut [0, 5, 6]);
  }
  assert_eq!(v, [0, 2, 3, 0, 5, 6]);
}

extern "C" {
  fn abs(input: i32) -> i32;
}

fn test_extern_c() {
  unsafe {
    println!("Abs val of -3: {}", abs(-3));
  }
}

static HELLO_WORLD: &str = "Hello World!";
static mut COUNTER: u32 = 0;

fn test_global() {
  println!("Global is: {}", HELLO_WORLD);
  unsafe {
    // Reading/Writing to globals needs unsafe
    COUNTER += 1;
    println!("COUNTER: {COUNTER}");
  }
}

union FloatUInt32 {
  f_u32: u32,
  f_f32: f32,
}

fn test_union() {
  let u1 = FloatUInt32 { f_f32: 1.0 };
  let u2 = FloatUInt32 { f_f32: 2.0 };
  let u4 = FloatUInt32 { f_f32: 4.0 };
  let u8 = FloatUInt32 { f_f32: 8.0 };
  let u16 = FloatUInt32 { f_f32: 16.0 };
  unsafe {
    println!("1.0 is {:#034b}", u1.f_u32);
    println!("2.0 is {:#034b}", u2.f_u32);
    println!("4.0 is {:#034b}", u4.f_u32);
    println!("8.0 is {:#034b}", u8.f_u32);
    println!("8.0 is {:#034b}", u16.f_u32);
  }
}

fn main() {
  test_raw_pointer();
  test_unsafe_function();
  test_unsafe_with_safe_wrapper();
  test_extern_c();
  test_global();
  test_union();
}
