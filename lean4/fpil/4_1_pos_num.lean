-- Positive Numbers
inductive Pos : Type where
  | one : Pos
  | succ : Pos → Pos

def seven : Pos :=
  Pos.succ (Pos.succ (Pos.succ (Pos.succ (Pos.succ (Pos.succ Pos.one)))))

-- Classes and Instances
class Plus (T : Type) where
  plus : T -> T -> T
instance : Plus Nat where
  plus := Nat.add
#eval Plus.plus 5 3

open Plus (plus)
#eval plus 5 3

def Pos.plus : Pos -> Pos -> Pos
  | Pos.one, k => Pos.succ k
  | Pos.succ m, k => Pos.succ (m.plus k)

instance : Plus Pos where
  plus := Pos.plus

def fourteen : Pos := plus seven seven

-- Overloaded Addition
instance : Add Pos where
  add := Pos.plus

def twentyone : Pos := seven + fourteen

-- Conversion to Strings
def posToString (atTop : Bool) (p : Pos) : String :=
  let paren s := if atTop then s else "(" ++ s ++ ")"
  match p with
    | Pos.one => "Pos.one"
    | Pos.succ n => paren s!"Pos.succ {posToString false n}"

instance : ToString Pos where
  toString := posToString true

#eval s!"There are {seven}"

-- Overloaded Multiplication
def Pos.mul : Pos -> Pos -> Pos
  | Pos.one, k => k
  | Pos.succ n, k => n.mul k + k

instance : Mul Pos where
  mul := Pos.mul

#eval [seven * Pos.one, seven * seven]

-- Literal Numbers
inductive LT3 where
  | zero
  | one
  | two
  deriving Repr

instance : OfNat LT3 0 where
  ofNat := LT3.zero

instance : OfNat LT3 1 where
  ofNat := LT3.one

instance : OfNat LT3 2 where
  ofNat := LT3.two

#eval (0 : LT3)
#eval (1 : LT3)
#eval (2 : LT3)
#eval (3 : LT3)

instance : OfNat Pos (n + 1) where
  ofNat :=
    let rec natPlusOne : Nat -> Pos
      | 0 => Pos.one
      | k + 1 => Pos.succ (natPlusOne k)
    natPlusOne n
#eval (8 : Pos)
#eval (0 : Pos)

-- Exercises
structure PosX where
  succ ::
  pred : Nat

instance : Add PosX where
  add :=
    let rec addPosX (n : PosX) (m : PosX) : PosX :=
      PosX.succ (n.pred + m.pred + 1)
    addPosX

instance : Mul PosX where
  mul :=
    let rec mulPosX (n : PosX) (m : PosX) : PosX :=
      PosX.succ (n.pred * m.pred + n.pred + m.pred)
    mulPosX

instance : ToString PosX where
  toString :=
    let toStringLoc (n : PosX) : String :=
      s!"PosX({n.pred + 1})"
    toStringLoc

instance : OfNat PosX (n + 1) where
  ofNat :=
    let rec natPlusOne : Nat -> PosX
      | 0 => PosX.succ 0
      | k + 1 => PosX.succ (k + 1)
    natPlusOne n

#eval (0 : PosX)
#eval (1 : PosX)
#eval (2 : PosX) + 5
#eval (2 : PosX) * 5

structure Even where
  double ::
  index : Nat

instance : Add Even where
  add :=
    let rec addEven (n : Even) (m : Even) : Even :=
      Even.double (n.index + m.index)
    addEven

instance : Mul Even where
  mul :=
    let rec mulEven (n : Even) (m : Even) : Even :=
      Even.double (2 * n.index * m.index)
    mulEven

instance : ToString Even where
  toString :=
    let toStringLoc (n : Even) : String :=
      s!"Even({2 * n.index})"
    toStringLoc

def two := Even.double 1
def four := Even.double 2
#eval two + four
#eval four * four
