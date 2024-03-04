-- Additional Conveniences
structure NonEmptyList (T : Type) : Type where
  head : T
  tail : List T
  deriving Repr

-- Constructor Syntax for Instances
structure TTree (T : Type) : Type where
  latinName : T
  commonNames : List T

def Tree := TTree String

def oak : Tree :=
  ⟨"Quercus robur", ["common oak", "European oak"]⟩

def birch : Tree := {
  latinName := "Betula pendula",
  commonNames := ["silver birch", "warty birch"]
}

def sloe : Tree where
  latinName := "Prunus spinosa"
  commonNames := ["sloe", "blackthorn"]

class Display (T : Type) where
  displayName : T → String

instance : Display Tree :=
  ⟨ TTree.latinName ⟩

instance : Display Tree := {
  displayName := TTree.latinName
}

instance : Display Tree where
  displayName t := t.latinName

-- Examples
example : NonEmptyList String := {
  head := "Sparrow",
  tail := ["Duck", "Swan", "Magpie", "Eurasian coot", "Crow"]
}

example (n : Nat) (k : Nat) : Bool :=
  n + k == k + n
