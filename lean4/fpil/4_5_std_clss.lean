-- Standard Classes
inductive Pos : Type where
  | one : Pos
  | succ : Pos → Pos
  deriving Repr

-- Arithmetic
#eval 5 + 2
#eval HAdd.hAdd 5 2
#eval 5 - 2
#eval HSub.hSub 5 2
#eval 5 * 2
#eval HMul.hMul 5 2
#eval 5 / 2
#eval HDiv.hDiv 5 2
#eval 5 % 2
#eval HMod.hMod 5 2
#eval 5 ^ 2
#eval HPow.hPow 5 2
#eval - 5
#eval Neg.neg 5

-- Bitwise Operators
#eval 12 &&& 10
#eval HAnd.hAnd 12 10
#eval 12 ||| 10
#eval HOr.hOr 12 10
#eval 12 ^^^ 10
#eval HXor.hXor 12 10
#eval ~~~ (15 : UInt8)
#eval Complement.complement (15 : UInt8)
#eval 4 >>> 1
#eval HShiftRight.hShiftRight 4 1
#eval 4 <<< 1
#eval HShiftLeft.hShiftLeft 4 1

-- Equality and Ordering
#eval "Octopus" == "Cuttlefish"
#eval BEq.beq "Octopus" "Cuttlefish"
#eval "Octopodes" == "Octo".append "podes"
#eval BEq.beq "Octopodes" ("Octo".append "podes")
#check 2 < 4
#check LT.lt 2 4
#eval 2 < 4
#eval LT.lt 2 4
#eval 2 <= 4
#eval LE.le 2 4
#eval 2 > 4
#eval GT.gt 2 4
#eval 2 >= 4
#eval GE.ge 2 4

def Pos.comp : Pos -> Pos -> Ordering
  | Pos.one, Pos.one => Ordering.eq
  | Pos.one, Pos.succ _ => Ordering.lt
  | Pos.succ _, Pos.one => Ordering.gt
  | Pos.succ n, Pos.succ m => comp n m

instance : Ord Pos where
  compare := Pos.comp

def one : Pos := Pos.one
def two : Pos := Pos.succ Pos.one

---- Does not work Ord -> LT/GT etc is not defined
#eval one < two

-- Hashing
def hashPos : Pos -> UInt64
  | Pos.one => 0
  | Pos.succ n => mixHash 1 (hashPos n)

instance : Hashable Pos where
  hash := hashPos

instance [Hashable T] : Hashable (List T) where
  hash xs := match xs with
    | [] => 0
    | x :: xr => mixHash (hash x) (hash xr)

#eval hash ([] : List Nat)

-- Deriving Standard Classes
inductive PosX : Type where
  | one : PosX
  | succ : PosX → PosX

deriving instance BEq, Hashable, Repr for PosX

-- Appending
structure NonEmptyList (T : Type) : Type where
  head : T
  tail : List T

deriving instance Repr for NonEmptyList

def idahoSpiders : NonEmptyList String := {
  head := "Banded Garden Spider",
  tail := [
    "Long-legged Sac Spider",
    "Wolf Spider",
    "Hobo Spider",
    "Cat-faced Spider"
  ]
}

instance : Append (NonEmptyList T) where
  append xs ys := { head := xs.head, tail := xs.tail ++ ys.head :: ys.tail }

#eval idahoSpiders ++ idahoSpiders

instance : HAppend (NonEmptyList T) (List T) (NonEmptyList T) where
  hAppend xs ys := { head := xs.head, tail := xs.tail ++ ys}

#eval idahoSpiders ++ ["Trapdoor Spider"]

-- Functors
#eval Functor.map (· + 5) [1, 2, 3]
#eval (· + 5) <$> [1, 2, 3]
#eval Functor.map toString (some [5, 6])
#eval toString <$> (some [5, 6])
#eval Functor.map List.reverse [[1, 2, 3], [4, 5, 6]]

instance : Functor NonEmptyList where
  map f xs := { head := f xs.head, tail := f <$> xs.tail }
