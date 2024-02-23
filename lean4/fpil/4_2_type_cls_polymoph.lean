-- Type Classes and Polymorphism
#check (IO.println)
#check @IO.println

-- Defining Polymorphic Functions with Instance Implicits
def List.sum [Add T] [OfNat T 0] : List T -> T
  | [] => 0
  | x :: xs => x + xs.sum

def fourNats : List Nat := [1, 2, 3, 4]
#eval fourNats.sum

structure PPoint (T : Type) where
  x : T
  y : T
  deriving Repr

instance { T : Type} [Add T] : Add (PPoint T) where
  add p1 p2 := { x := p1.x + p2.x, y := p1.y + p2.y }

def point12 : PPoint Nat := { x := 1, y := 2 }
def point23 : PPoint Nat := { x := 2, y := 3 }

#eval point12 + point23

-- Methods and Implicit Arguments
#check OfNat.ofNat
#check (OfNat.ofNat)
#check @OfNat.ofNat
#check @HAdd.hAdd


-- Exercises
structure Even where
  ctor ::
  num : Nat
  deriving Repr

instance : OfNat Even 0 where
  ofNat := Even.ctor 0

instance [OfNat Even n] : OfNat Even (Nat.succ (Nat.succ n)) where
  ofNat := Even.ctor (n + 2)

#eval (0 : Even)
#eval (1 : Even)
#eval (2 : Even)

#eval (512 : Even)
#eval (256 : Even)
#eval (128 : Even)
#eval (192 : Even)
#eval (224 : Even)
#eval (240 : Even)
#eval (248 : Even)
#eval (252 : Even)
#eval (254 : Even)
