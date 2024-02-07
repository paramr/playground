-- Additional Conveniences

-- Automatic Implicit Arguments
def length {T : Type} (xs : List T) : Nat :=
  match xs with
    | [] => 0
    | _ :: ys => Nat.succ (length ys)

def lengthI (xs : List T) : Nat :=
  match xs with
    | [] => 0
    | _ :: ys => Nat.succ (lengthI ys)

-- Pattern-Matching Definitions
def lengthPM : List T -> Nat
  | [] => 0
  | _ :: ys => Nat.succ (lengthPM ys)

def drop : Nat -> List T -> List T
  | Nat.zero, l => l
  | _, [] => []
  | Nat.succ n, _ :: lt => drop n lt

def Option.fromOption (default : T) : Option T -> T
  | none => default
  | some x => x

#eval (some 5).fromOption 2
#eval none.fromOption 2

-- Local Definitions
def unzipNaive : List (U × V) -> List U × List V
  | [] => ([], [])
  | (x, y) :: xys => (x :: (unzipNaive xys).fst, y :: (unzipNaive xys).snd)

def unzipLet : List (U × V) -> List U × List V
  | [] => ([], [])
  | (x, y) :: xys =>
    let unzipped : List U × List V := unzipLet xys
    (x :: unzipped.fst, y :: unzipped.snd)

def unzipLetPM : List (U × V) -> List U × List V
  | [] => ([], [])
  | (x, y) :: xys =>
    let (xs, ys) : List U × List V := unzipLet xys
    (x :: xs, y :: ys)

def reverse (l : List T) : List T :=
  let rec helper : List T -> List T -> List T
    | [], soFar => soFar
    | y :: ys, soFar => helper ys (y :: soFar)
  helper l []

-- Type Inference
def unzipTI : List (U × V) -> List U × List V
  | [] => ([], [])
  | (x, y) :: xys =>
    let unzipped := unzipTI xys
    (x :: unzipped.fst, y :: unzipped.snd)

def unzipRTI (pairs : List (U × V)) :=
  match pairs with
    | [] => ([], [])
    | (x, y) :: xys =>
      let unzipped := unzipRTI xys
      (x :: unzipped.fst, y :: unzipped.snd)

def idE (x : T) : T := x
def idRI (x : T) := x

-- Simultaneous Matching
def dropX (n : Nat) (xs : List T) : List T :=
  match n, xs with
    | Nat.zero, ys => ys
    | _, [] => []
    | Nat.succ n, _ :: ys => dropX n ys

-- Natural Number Patterns
def even : Nat -> Bool
  | 0 => true
  | n + 1 => not (even n)

def halvePeano : Nat -> Nat
  | Nat.zero => 0
  | Nat.succ Nat.zero => 0
  | Nat.succ (Nat.succ n) => halvePeano n + 1

def halve : Nat -> Nat
  | 0 => 0
  | 1 => 0
  | n + 2 => halve n + 1

-- Anonymous Functions
#check fun x => x + 1
#check fun (x : Int) => x + 1
def test_fun := fun x => x + 1
#check test_fun
#check fun {T : Type} (x : T) => x

#check fun
  | 0 => none
  | n + 1 => some n

def double : Nat -> Nat := fun
  | 0 => 0
  | k + 1 => double k + 2

#check (· + 1)
#eval (· * 2) 5

-- Namespaces
def Nat.double (x : Nat) : Nat := x + x
#eval (4 : Nat).double

namespace NewNamespace
def triple (x : Nat) := 3 * x
def quadruple := fun x => 4 * x
end NewNamespace

def timesTwelve (x : Nat) :=
  open NewNamespace in
  quadruple (triple x)

open NewNamespace in
#check quadruple

-- if let
inductive Inline : Type where
  | lineBreak
  | string : String -> Inline
  | emph : Inline -> Inline
  | strong : Inline -> Inline

def Inline.stringM? (inline : Inline) : Option String :=
  match inline with
    | Inline.string s => some s
    | _ => none

def Inline.stringIL? (inline : Inline) : Option String :=
  if let Inline.string s := inline then
    some s
  else
    none

-- Positional Structure Arguments
structure Point where
  x : Float
  y: Float
  deriving Repr

#eval (⟨1, 2⟩ : Point)

-- String Interpolation
#eval s!"three fives is {NewNamespace.triple 5}"
-- #check s!"three fives is {NewNamespace.triple}"
