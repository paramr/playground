-- Datatypes and Patterns
inductive XBool where
  | false : XBool
  | true : XBool

#check XBool.false

inductive XNat where
  | zero : XNat
  | succ (n : XNat) : XNat

#check XNat.succ

-- Pattern Matching
def isZero (n : Nat) : Bool :=
  match n with
  | Nat.zero => true
  | Nat.succ _ => false

#eval isZero 0
#eval isZero 5

def pred (n : Nat) : Nat :=
  match n with
  | Nat.zero => Nat.zero
  | Nat.succ k => k

#eval pred 5
#eval pred 0

structure Point3D where
  x : Float
  y : Float
  z : Float
  deriving Repr

def depth (p : Point3D) : Float :=
  match p with
    | { x := _, y := _, z := d } => d

-- Recursive Functions
def even (n : Nat) : Bool :=
  match n with
    | Nat.zero => true
    | Nat.succ k => not (even k)

def plus (n : Nat) (m : Nat) : Nat :=
  match m with
    | Nat.zero => n
    | Nat.succ m' => Nat.succ (plus n m')

def times (n : Nat) (m : Nat) : Nat :=
  match m with
    | Nat.zero => Nat.zero
    | Nat.succ m' => plus (times n m') n

def minus (n : Nat) (m : Nat) : Nat :=
  match m with
    | Nat.zero => n
    | Nat.succ m' => pred (minus n m')

-- def div (n : Nat) (m : Nat) : Nat :=
--   if n < m then
--     0
--   else
--     Nat.succ (div (n - m) m)
