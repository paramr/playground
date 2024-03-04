-- Coercions
inductive Pos : Type where
  | one : Pos
  | succ : Pos → Pos
  deriving Repr

def pos_two : Pos := Pos.succ Pos.one

def Pos.toNat : Pos -> Nat
  | one => 1
  | succ n => n.toNat + 1

#eval [1, 2, 3, 4].drop pos_two

instance : Coe Pos Nat where
  coe := Pos.toNat

#eval [1, 2, 3, 4].drop pos_two
#check [1, 2, 3, 4].drop pos_two

-- Chaining Coercions
--- Pos -> Nat -> Int
def oneInt : Int := Pos.one

--- Cycles are ok in resolution.
inductive A where
  | a

inductive B where
  | b

instance : Coe A B where
  coe _ := B.b

instance : Coe B A where
  coe _ := A.a

instance : Coe Unit A where
  coe _ := A.a

def coercedToB : B := ()

--- Coersions should be "consistent" in some sense. Also seems like
--- similar to Univalence but one sided. i.e if T can be coerced into U
--- then a theorem about T can be moved to U? Nat can be coerced into
--- reals but are all theorems in number theory valid for reals?


def List.last? : List T -> Option T
  | [] => none
  | [x] => x
  | _ :: x :: xs => last? (x :: xs)

def ppp : Option (Option (Option String)) := "Something"
def pppN : Option (Option (Option Int)) := 392
def pppN1 : Option (Option (Option Int)) := (392 : Nat)

#check 392

-- Non-Empty Lists and Dependent Coercions
structure NonEmptyList (T : Type) : Type where
  head : T
  tail : List T
  deriving Repr

instance : Coe (NonEmptyList T) (List T) where
  coe
    | { head := x, tail := xs } => x :: xs

def idahoSpiders : NonEmptyList String := {
  head := "Banded Garden Spider",
  tail := [
    "Long-legged Sac Spider",
    "Wolf Spider",
    "Hobo Spider",
    "Cat-faced Spider"
  ]
}

#eval List.reverse idahoSpiders
#eval (List.reverse idahoSpiders : List String)
#eval List.reverse (α := String) idahoSpiders

instance : CoeDep (List T) (x :: xs) (NonEmptyList T) where
  coe := { head := x, tail := xs }

def nel1 : NonEmptyList Nat := [1, 2, 3]
def l1 := [1, 2, 3]
def nel2 : NonEmptyList Nat := l1

-- Coercing to Types
structure Monoid where
  Carrier : Type
  neutral : Carrier
  op : Carrier -> Carrier -> Carrier

def natMulMonoid : Monoid :=
  { Carrier := Nat, neutral := 1, op := Nat.mul }

def natAddMonoid : Monoid :=
  { Carrier := Nat, neutral := 0, op := Nat.add }

def stringMonoid : Monoid :=
  { Carrier := String, neutral := "", op := String.append }

def listMonoid (T : Type) : Monoid :=
  { Carrier := List T, neutral := [], op := List.append }

def foldMapE (M : Monoid) (f : T -> M.Carrier) (xs : List T) : M.Carrier :=
  let rec go (soFar : M.Carrier) : List T -> M.Carrier
    | [] => soFar
    | y :: ys => go (M.op soFar (f y)) ys
  go M.neutral xs

instance : CoeSort Monoid Type where
  coe m := m.Carrier

def foldMap (M : Monoid) (f : T -> M) (xs : List T) : M :=
  let rec go (soFar : M) : List T -> M
    | [] => soFar
    | y :: ys => go (M.op soFar (f y)) ys
  go M.neutral xs

-- Coercing to Functions
structure Adder where
  howMuch : Nat

def add5 : Adder := { howMuch := 6 }

instance : CoeFun Adder (fun _ => Nat -> Nat) where
  coe a := (. + a.howMuch)

#eval add5 3
