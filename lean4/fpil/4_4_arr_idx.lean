-- Arrays and Indexing

-- Arrays
def northernTrees : Array String :=
  #["sloe", "birch", "elm", "oak"]
#eval northernTrees.size

-- Non-Empty Lists
structure NonEmptyList (T : Type) : Type where
  head : T
  tail : List T

def idahoSpiders : NonEmptyList String := {
  head := "Banded Garden Spider",
  tail := [
    "Long-legged Sac Spider",
    "Wolf Spider",
    "Hobo Spider",
    "Cat-faced Spider"
  ]
}

def NonEmptyList.get? : NonEmptyList T -> Nat -> Option T
  | xs, 0 => some xs.head
  | { head := _, tail := []}, _ + 1 => none
  | { head := _, tail := h :: t}, n + 1 => get? { head := h, tail := t } n

def NonEmptyList.getX? : NonEmptyList T -> Nat -> Option T
  | xs, 0 => some xs.head
  | xs, n + 1 => xs.tail.get? n

#eval idahoSpiders.get? 1
#eval idahoSpiders.getX? 4
#eval idahoSpiders.getX? 5

abbrev NonEmptyList.inBounds (xs : NonEmptyList T) (i : Nat) : Prop :=
  i <= xs.tail.length

theorem atLeastThreeSpiders : idahoSpiders.inBounds 2 := by trivial
theorem notSixSpiders : !idahoSpiders.inBounds 5 := by trivial
theorem sixSpiders : idahoSpiders.inBounds 5 := by trivial

def NonEmptyList.get (xs : NonEmptyList T) (i : Nat) (_ : xs.inBounds i) : T :=
  match i with
    | 0 => xs.head
    | n + 1 => xs.tail[n]

#eval idahoSpiders.get 1 (by trivial)
#eval idahoSpiders.get 4 (by trivial)
-- Crashed?!? should not work
-- #eval idahoSpiders.get 4 (by trivial)

-- Overloading Indexing
instance : GetElem (NonEmptyList T) Nat T NonEmptyList.inBounds where
  getElem := NonEmptyList.get

#eval idahoSpiders[1]

structure PPoint (T : Type) where
  x : T
  y : T
  deriving Repr

instance : GetElem (PPoint T) Bool T (fun _ _ => True) where
  getElem (p : PPoint T) (i : Bool) _ :=
    if i then p.y else p.x
