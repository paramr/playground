-- Dependent Type Theory

-- Simple Type Theory
def m : Nat := 1
def n : Nat := 0
def b1 : Bool := true
def b2 : Bool := false

#check m
#check n
#check n + 0
#check m * (n + 0)
#check b1
#check b1 && b2
#check b1 || b2
#check true

#eval 5 * 4
#eval m + 2
#eval b1 && b2

#check Nat → Nat
#check Nat -> Nat
#check Nat × Nat
#check Prod Nat Nat
#check Nat → Nat → Nat
#check Nat → (Nat → Nat)
#check Nat × Nat → Nat
#check (Nat → Nat) → Nat
#check Nat.succ
#check (0, 1)
#check Nat.add
#check Nat.succ 2
#check Nat.add 3
#check (5, 9).1
#check (5, 9).2

#eval Nat.succ 2
#eval Nat.add 5 2
#eval (5, 9).1
#eval (5, 9).2

-- Types as objects
#check Nat
#check Bool
#check Nat → Bool
#check Nat × Bool
#check Nat → Nat
#check Nat × Nat → Nat
#check Nat → Nat → Nat
#check Nat → Nat → Bool
#check (Nat → Nat) → Nat

def α : Type := Nat
def β : Type := Bool
def F : Type → Type := List
def G : Type → Type → Type := Prod

#check α
#check F α
#check F Nat
#check G α
#check G α β
#check G α Nat

#check Prod α β
#check α × β
#check Prod Nat Nat
#check Nat × Nat

#check List α
#check List Nat

#check Type
#check Type 1
#check Type 2
#check Type 32
#check Type 33
#check Prop
#check True
#check trivial

#check List
#check Prod

universe u

def F1 (α : Type u) : Type u := Prod α α
#check F1

def F2.{v} (α : Type v) : Type v := Prod α α
#check F2

-- Function Abstraction and Evaluation
#check fun (x : Nat) => x + 5
#check λ (x : Nat) => x + 5
#check fun x => x + 5
#check λ x => x + 5
#eval (λ x : Nat => x + 5) 10

#check fun x : Nat => fun y : Bool => if not y then x + 1 else x + 2
#check fun (x : Nat) (y : Bool) => if not y then x + 1 else x + 2
#check fun x y => if not y then x + 1 else x + 2   -- Nat → Bool → Nat

def f (n : Nat) : String := toString n
def g (s : String) : Bool := s.length > 0
#check fun x : Nat => x
#check fun _ : Nat => true
#check fun x : Nat => g (f x)
#check fun x => g (f x)

#check fun (g : String → Bool) (f : Nat → String) (x : Nat) => g (f x)
#check fun (α β γ : Type) (g : β → γ) (f : α → β) (x : α) => g (f x)

#check (fun x : Nat => x) 1
#check (fun _ : Nat => true) 1
#check (fun (α β γ : Type) (u : β → γ) (v : α → β) (x : α) => u (v x)) Nat String Bool g f 0

#eval (fun x : Nat => x) 1
#eval (fun _ : Nat => true) 1

-- Definitions
def double := fun (x : Nat) => x + x
#eval double 3

def pi := 3.141592654

def add (x y : Nat) := x + y
#eval add 3 2
#eval add (double 3) (7 + 9)

def greater (x y : Nat) := if x > y then x else y
#eval greater 5 3

def doTwice (f : Nat → Nat) (x : Nat) : Nat := f (f x)
#eval doTwice double 2

def compose (α β γ : Type) (g : β → γ) (f : α → β) (x : α) : γ := g (f x)

def square (x : Nat) : Nat := x * x
#eval (compose Nat Nat Nat square double) 5

-- Local Definitions
#check let y := 2; y + y
#eval let y := 2; y + y

def twice_double (x : Nat) : Nat :=
  let y := x + x
  y * y
#eval twice_double 2

#check let y := 2 + 2; let z := y + y; z * z
#eval  let y := 2 + 2; let z := y + y; z * z

def foo := let a := Nat; fun x : a => x + 2
def bar := (fun a => fun x : a => x + 2) Nat

-- Variables and Sections
variable (α β γ : Type)
section useful
  variable (g : β → γ) (f : α → β) (h : α → α)
  variable (x : α)

  def composev := g (f x)

  def doTwicev := h (h x)

  def doThricev := h (h (h x))
  #check x
end useful

#check x
#check α
#print composev
#print doTwicev
#print doThricev

-- Namespaces
namespace Foo
  def a : Nat := 5
  def f (x : Nat) : Nat := x + 7
  def fa : Nat := f a
  namespace Bar
    def ffa : Nat := f (f a)
    #check fa
    #check ffa
  end Bar
  #check a
  #check f
  #check fa
  #check Foo.fa
  #check Bar.ffa
  #check Foo.Bar.ffa
end Foo

#check Foo.a
#check Foo.f
#check Foo.fa
#check Foo.Bar.ffa

section Test
  open Foo
  #check a
  #check f
  #check fa
  #check Foo.fa
  #check Bar.ffa
end Test

#check a

#check List.nil
#check List.cons
#check List.map

open List

#check nil
#check cons
#check map

namespace Foo
  def fffa : Nat := f (f (f (f a)))
end Foo

-- What makes dependent type theory dependent?
#check nil (α := Nat)
#check cons (α := Nat)
#check @List.nil
#check @List.cons

def f1 (α : Type u) (β : α → Type v) (a : α) (b : β a) : (a : α) × β a :=
  ⟨a, b⟩

def g1 (α : Type u) (β : α → Type v) (a : α) (b : β a) : Σ a : α, β a :=
  Sigma.mk a b

def h1 (x : Nat) : Nat :=
  (f1 Type (fun α => α) Nat x).2

#eval h1 5

def h2 (x : Nat) : Nat :=
  (g1 Type (fun α => α) Nat x).2

#eval h2 5

-- Implicit Arguments
def ident {α : Type u} (x : α) := x

#check ident
#check ident 1
#check ident "1"
#check @ident

section
  variable {α : Type u}
  variable (x : α)
  def ident1 := x
end

#check ident1
#check ident1 4
#check ident1 "hello"

#check List.nil
#check id
#check (List.nil : List Nat)
#check (id : Nat → Nat)
#check 2
#check (2 : Nat)
#check (2 : Int)
#check id 2
#check @id 2
#check @id Nat 2
