-- Controlling Instance Search
inductive Pos : Type where
  | one : Pos
  | succ : Pos → Pos
  deriving Repr

instance : OfNat Pos (n + 1) where
  ofNat :=
    let rec natPlusOne : Nat -> Pos
      | 0 => Pos.one
      | k + 1 => Pos.succ (natPlusOne k)
    natPlusOne n

def addNatPos : Nat -> Pos -> Pos
  | 0, p => p
  | n + 1, p => Pos.succ (addNatPos n p)

def addPosNat : Pos -> Nat -> Pos
  | p, 0 => p
  | p, n + 1 => Pos.succ (addPosNat p n)

-- Heterogeneous Overloadings
instance : HAdd Nat Pos Pos where
  hAdd := addNatPos

instance : HAdd Pos Nat Pos where
  hAdd := addPosNat

#eval (3 : Pos) + (5 : Nat)
#eval (3 : Nat) + (5 : Pos)

-- Output Parameters
class HPlus (T : Type) (U : Type) (V : Type) where
  hPlus : T → U → V

instance : HPlus Nat Pos Pos where
  hPlus := addNatPos

instance : HPlus Pos Nat Pos where
  hPlus := addPosNat

#eval HPlus.hPlus (3 : Pos) (5 : Nat)
#eval (HPlus.hPlus (3 : Pos) (5 : Nat) : Pos)

class HPlusX (T : Type) (U : Type) (V : outParam Type) where
  hPlus : T → U → V

instance : HPlusX Nat Pos Pos where
  hPlus := addNatPos

instance : HPlusX Pos Nat Pos where
  hPlus := addPosNat

#eval HPlusX.hPlus (3 : Pos) (5 : Nat)
#eval (HPlusX.hPlus (3 : Pos) (5 : Nat) : Pos)

-- Default Instances
instance [Add T] : HPlusX T T T where
  hPlus := Add.add

#eval HPlusX.hPlus (3 : Nat) (5 : Nat)
#check HPlusX.hPlus (5 : Nat) (3 : Nat)

#check HPlusX.hPlus (5 : Nat)

@[default_instance]
instance [Add T] : HPlusX T T T where
  hPlus := Add.add

#check HPlusX.hPlus (5 : Nat)

-- Exercises
structure PPoint (T : Type) where
  x : T
  y : T
  deriving Repr

instance [Mul T] : HMul (PPoint T) T (PPoint T) where
  hMul := fun (p : PPoint T) (s : T) => { x := p.x * s, y := p.y * s }

#eval {x := 2.5, y := 3.7 : PPoint Float} * 2.0
