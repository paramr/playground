structure PPoint (T : Type) where
  x : T
  y : T
  deriving Repr

def natOrigin : PPoint Nat :=
  { x := Nat.zero, y := 0 }

def replaceX (T : Type) (point : PPoint T) (newX : T) : PPoint T :=
  { point with x := newX }

#check (replaceX)
#check replaceX Nat
#check replaceX Nat natOrigin
#check replaceX Nat natOrigin 5
#eval replaceX Nat natOrigin 5

inductive Sign where
  | pos
  | neg

def posOrNegThree (s : Sign) : match s with | Sign.pos => Nat | Sign.neg => Int :=
  match s with
    | Sign.pos => (3 : Nat)
    | Sign.neg => (-3 : Int)

#eval posOrNegThree Sign.pos
#eval posOrNegThree Sign.neg

-- Linked Lists
def primesUnder10 : List Nat :=
  [2, 3, 5, 7]

inductive XList (T : Type) where
  | nil : XList T
  | cons : T -> XList T -> XList T

def explicitPrimesUnder10 : List Nat :=
  List.cons 2
    (List.cons 3
      (List.cons 5
        (List.cons 7 List.nil)))
#eval explicitPrimesUnder10

def length (T : Type) (l : XList T) : Nat :=
  match l with
    | XList.nil => Nat.zero
    | XList.cons _ lt => Nat.succ (length T lt)

def List.lengthP (T : Type) (el: Nat) (l : List T) : Nat :=
  match l with
    | [] => el
    | _lh :: lt => 1 + lengthP T el lt

#eval primesUnder10.lengthP 5

-- Implicit Arguments
def replaceXImp {T : Type} (point : PPoint T) (newX : T) : PPoint T :=
  { point with x := newX }

#eval replaceXImp natOrigin 5

def List.lengthImp {T : Type} (l : List T) : Nat :=
  match l with
    | [] => 0
    | _ :: lt => Nat.succ (lengthImp lt)

#eval primesUnder10.lengthImp

#check List.lengthImp (T := Int)

-- More Built-In Datatypes

--- Option
inductive XOption (T : Type) where
  | none : XOption T
  | some (val : T) : XOption T

def List.headX? {T : Type} {l : List T} : Option T :=
  match l with
    | [] => none
    | y :: _ => some y

#eval primesUnder10.headX?

#eval [].headX? (T := Nat)
#eval ([] : List Int).head?

--- Prod
structure XProd (U : Type) (V : Type) : Type where
  fst : U
  snd : V

def fives : String × Int := { fst := "five", snd := 5 }
def fivesS : String × Int := ("five", 5)

def sevens : String × Int × Nat := ("VII", 7, 4 + 3)
def sevensA : String × (Int × Nat) := ("VII", (7, 4 + 3))

--- Sum
inductive  XSum (U : Type) (V : Type) where
  | inl : U -> XSum U V
  | inr : V -> XSum U V

def PetName : Type := String ⊕ String
def animals : List PetName :=
  [Sum.inl "Spot", Sum.inr "Tiger", Sum.inl "Fifi", Sum.inl "Rex", Sum.inr "Floof" ]

def howManyDogs (pets : List PetName) : Nat :=
  match pets with
    | [] => 0
    | Sum.inl _ :: morePets => howManyDogs morePets + 1
    | Sum.inr _ :: morePets => howManyDogs morePets

--- Unit
inductive XUnit : Type where
| unit : XUnit

--- Empty
inductive XEmpty : Type

-- Exercises
def lastElement {T : Type} (l : List T) : Option T :=
  match l with
    | [] => none
    | lh :: lt =>
      match lastElement lt with
        | none => some lh
        | some t => some t

#eval lastElement (T := Int) []

def List.findFirst? {T : Type} (l : List T) (pred : T -> Bool) : Option T :=
  match l with
    | [] => none
    | lh :: lt => if pred lh then some lh else lt.findFirst? pred

#eval primesUnder10.findFirst? (fun x => x >= 5)
#eval primesUnder10.findFirst? (fun x => x >= 10)

def Prod.swap {U V : Type} (pair : U × V) : V × U :=
  (pair.snd, pair.fst)
#eval (1, 2).swap

inductive PetNameI where
  | dog : String -> PetNameI
  | cat : String -> PetNameI

def txPets (l : List PetName) : List PetNameI :=
  match l with
    | [] => []
    | lh :: lt =>
      match lh with
        | Sum.inl n => PetNameI.dog n :: txPets lt
        | Sum.inr n => PetNameI.cat n :: txPets lt

def zip {U V : Type} (xs : List U) (ys : List V) : List (U × V) :=
  match xs with
    | [] => []
    | xh :: xt =>
      match ys with
        | [] => []
        | yh :: yt => (xh, yh) :: zip xt yt

#eval zip [1, 2, 3, 4] [1, 4, 9, 16]

def take {T : Type} (n : Nat) (l : List T) :=
  match n with
    | Nat.zero => []
    | Nat.succ np =>
      match l with
        | [] => []
        | lh :: lt => lh :: take np lt
#eval take 3 ["bolete", "oyster"]
#eval take 1 ["bolete", "oyster"]

def dist {U V W : Type} (inp: U × (V ⊕ W)) : (U × V) ⊕ (U × W) :=
  match inp.snd with
    | Sum.inl v => Sum.inl (inp.fst, v)
    | Sum.inr w => Sum.inr (inp.fst, w)
#eval dist (U := Nat) (V := Nat) (W := Nat) (5, Sum.inl 2)

def boolToSum {T : Type} (b : Bool) (t : T) : T ⊕ T :=
  match b with
    | false => Sum.inl t
    | true => Sum.inr t
#eval boolToSum false 1
#eval boolToSum true 2
