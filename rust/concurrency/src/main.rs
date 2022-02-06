use std::thread;
use std::time::Duration;

fn test_thread_early_exit() {
  println!("test_thread_early_exit");
  let _handle = thread::spawn(|| {
    for i in 1..10 {
      println!("  Early Spwned: {i}");
      thread::sleep(Duration::from_millis(1))
    }
  });
  for i in 1..5 {
    println!("  Early Main: {i}");
    thread::sleep(Duration::from_millis(1))
  }
}

fn test_thread_join() {
  println!("test_thread_join");
  let handle = thread::spawn(|| {
    for i in 1..10 {
      println!("  Join Spwned: {i}");
      thread::sleep(Duration::from_millis(1))
    }
  });
  for i in 1..5 {
    println!("  Join Main: {i}");
    thread::sleep(Duration::from_millis(1))
  }
  handle.join().unwrap();
}

fn test_thread_closure() {
  println!("test_thread_closure");
  let v = vec![[1, 2, 3]];
  let handle = thread::spawn(move || {
    println!("  Spawn Vector: {v:?}");
  });
  // v got moved
  // println!("v is {v:?}");
  // drop(v);
  handle.join().unwrap();
}

use std::sync::mpsc;

fn create_senders(
  tx: mpsc::Sender<String>,
  list: Vec<String>,
) -> thread::JoinHandle<()> {
  return thread::spawn(move || {
    for val in list {
      tx.send(val).unwrap();
      thread::sleep(Duration::from_millis(300));
    }
  });
}

fn test_mpsc_channel() {
  let (tx, rx) = mpsc::channel();
  let mut join_handles = Vec::new();

  join_handles.push(create_senders(
    tx.clone(),
    vec![
      String::from("hello"),
      String::from("from"),
      String::from("the"),
      String::from("thread"),
    ],
  ));
  join_handles.push(create_senders(
    tx.clone(),
    vec![
      String::from("elloh"),
      String::from("romf"),
      String::from("het"),
      String::from("hreadt"),
    ],
  ));
  drop(tx);
  for val in rx {
    println!("Got: {val}");
  }
  for handle in join_handles {
    handle.join().unwrap();
  }
}

use std::sync::{Arc, Mutex};

fn test_mutex() {
  let mutex = Arc::new(Mutex::new(0));
  {
    let mut num = mutex.lock().unwrap();
    *num = 1;
  }
  println!("m = {mutex:?}");

  let mut join_handles = Vec::new();
  for _ in 1..10 {
    let mutex = Arc::clone(&mutex);
    join_handles.push(thread::spawn(move || {
      let mut num = mutex.lock().unwrap();
      *num += 1;
    }));
  }
  for handle in join_handles {
    handle.join().unwrap();
  }
  println!("Result: {mutex:?}");
}

fn main() {
  test_thread_early_exit();
  test_thread_join();
  test_thread_closure();
  test_mpsc_channel();
  test_mutex();
}
