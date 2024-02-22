-- Functions and Definitions
def hello := "Hello"

def lean : String := "Lean"

#eval String.append hello (String.append " " lean)

-- Defining Functions
def add1 (n : Nat) : Nat := n + 1

#check add1
#check (add1)

#eval add1 7

def maximum (n : Nat) (k : Nat) : Nat :=
  if n < k then
    k
  else
    n

#check maximum
#check (maximum)
#check maximum 3

#eval maximum (5 + 8) (2 * 7)

#check String.append "Hello "

-- Exercises
def joinStringsWith (c f s: String): String :=
  String.append f (String.append c s)

#check joinStringsWith
#eval joinStringsWith ", " "one" "and another"
#check joinStringsWith ": "

def volume (l b h : Nat): Nat :=
  l * b * h

#check volume
#eval volume 2 3 4

-- Defining Types
def Str : Type := String
def aStr : Str := "This is a string."

-- Errors
def NaturalNumber : Type := Nat
def thirtyEightDirect : NaturalNumber := 38
def thirtyEight : NaturalNumber := (38 : Nat)

abbrev N : Type := Nat
def thirtyNine : N := 39
#eval thirtyNine
