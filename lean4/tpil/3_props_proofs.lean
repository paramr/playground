-- Propositions and Proofs

-- Propositions as Types
section pat
def Implies (p q : Prop) : Prop := p → q

#check And
#check Or
#check Not
#check Implies

variable (p q r : Prop)
#check And p q
#check Or (And p q) r
#check Implies (And p q) (And q p)

structure Proof (p : Prop) : Type where
  proof : p
#check Proof

axiom and_comm (p q : Prop) :  Proof (Implies (And p q) (And q p))
#check and_comm p q

axiom modus_ponens : (p q : Prop) → Proof (Implies p q) → Proof p → Proof q
axiom implies_intro : (p q : Prop) → (Proof p → Proof q) → Proof (Implies p q)
end pat

-- Working with Propositions as Types
section wpat
variable {p q r s : Prop}

theorem t1 : p → q → p := fun hp : p => fun _ : q => hp
#check t1
#print t1

theorem t1_1 : p → q → p :=
  fun hp : p =>
  fun _ : q =>
  show p from hp
#print t1_1

theorem t1_2 (hp : p) (_ : q) : p := hp
#print t1_2

axiom hp : p
#check hp
theorem t2 : q → p := t1 hp

axiom unsound : False
theorem ex : 1 = 0 := False.elim unsound

theorem t1_3 {p q : Prop} (hp : p) (_ : q) : p := hp
#print t1_3

theorem t1_4 : ∀ {p q : Prop}, p → q → p :=
  fun {p q : Prop} (hp : p) (_ : q) => hp
#print t1_4

theorem t1_5 : p → q → p := fun (hp : p) (_ : q) => hp
#print t1_5

variable (hp : p)
theorem t1_6 : q → p := fun (_ : q) => hp
#print t1_6

theorem t1e (p q : Prop) (hp : p) (_ : q) : p := hp
#check t1e p q
#check t1e r s
#check t1e (r → s) (s → r)

variable (h : r → s)
#check t1e (r → s) (s → r) h

theorem t3 (h₁ : q → r) (h₂ : p → q) : p → r :=
  fun h₃ : p =>
  show r from h₁ (h₂ h₃)
#check t3
end wpat

-- Propositional Logic
section pl
variable (p q r: Prop)
variable (hp : p) (hq : q)


#check p → q → p ∧ q
#check ¬p → p ↔ False
#check p ∨ q → q ∨ p

--- Conjunction
example (hp : p) (hq : q) : p ∧ q := And.intro hp hq
#check fun (hp : p) (hq : q) => And.intro hp hq

example (h : p ∧ q) : p := And.left h
example (h : p ∧ q) : q := And.right h
example (h : p ∧ q) : q ∧ p :=
  And.intro (And.right h) (And.left h)

#check (⟨hp, hq⟩ : p ∧ q)

example (h : p ∧ q) : q ∧ p :=
  ⟨h.right, h.left⟩

example (h : p ∧ q) : q ∧ p ∧ q :=
  ⟨h.right, ⟨h.left, h.right⟩⟩

example (h : p ∧ q) : q ∧ p ∧ q :=
  ⟨h.right, h.left, h.right⟩

axiom hpqr : p ∧ q ∧ r
#check (hpqr p q r).right

--- Disjunction
example (hp : p) : p ∨ q := Or.intro_left q hp
example (hq : q) : p ∨ q := Or.intro_right p hq

example (h : p ∨ q) : q ∨ p :=
  Or.elim h
    (fun hp : p =>
      show q ∨ p from Or.intro_right q hp)
    (fun hq : q =>
      show q ∨ p from Or.intro_left p hq)

example (h : p ∨ q) : q ∨ p :=
  Or.elim h
    (fun hp => Or.inr hp)
    (fun hq => Or.inl hq)

example (h : p ∨ q) : q ∨ p :=
  h.elim (fun hp => Or.inr hp) (fun hq => Or.inl hq)

--- Negation and Falsity
example (hpq : p → q) (hnq : ¬q) : ¬p :=
  fun hp : p =>
    show False from hnq (hpq hp)

example (hp : p) (hnp : ¬p) : q := False.elim (hnp hp)
example (hp : p) (hnp : ¬p) : q := absurd hp hnp
example (hnp : ¬p) (hq : q) (hqp : q → p) : r :=
  absurd (hqp hq) hnp

--- Logical Equivalence
theorem and_swap : p ∧ q ↔ q ∧ p :=
  Iff.intro
    (fun h : p ∧ q =>
     show q ∧ p from And.intro (And.right h) (And.left h))
    (fun h : q ∧ p =>
     show p ∧ q from And.intro (And.right h) (And.left h))
#check and_swap p q

variable (h : p ∧ q)
example : q ∧ p := (and_swap p q).mp h

theorem and_swap_2 : p ∧ q ↔ q ∧ p :=
  ⟨ fun h => ⟨h.right, h.left⟩, fun h => ⟨h.right, h.left⟩ ⟩
example (h : p ∧ q) : q ∧ p := (and_swap_2 p q).mp h

end pl

-- Introducing Auxiliary Subgoals
section subg
variable (p q : Prop)

example (h : p ∧ q) : q ∧ p :=
  have hp : p := h.left
  have hq : q := h.right
  show q ∧ p from And.intro hq hp

example (h : p ∧ q) : q ∧ p :=
  have hp : p := h.left
  suffices hq : q from And.intro hq hp
  show q from And.right h

end subg

-- Classical Logic
section cl
open Classical
variable (p : Prop)

#check em p

theorem dne {p : Prop} (h : ¬¬p) : p :=
  Or.elim (em p)
    (fun hp : p => hp)
    (fun hnp : ¬p => absurd hnp h)

theorem dne_to_em (hdne : {p : Prop} -> ¬¬p → p) : {p : Prop} -> p ∨ ¬p :=
  fun {p : Prop} =>
    suffices hnn_ponp : ¬¬(p ∨ ¬p) from hdne hnn_ponp
    show ¬¬(p ∨ ¬p) from
      fun (hn_ponp : ¬(p ∨ ¬p)) =>
        have hnp : ¬p := fun (hp : p) =>
          hn_ponp (Or.intro_left (¬p) hp)
        hn_ponp (Or.intro_right p hnp)

example {p : Prop} (h : ¬¬p) : p :=
  byCases
    (fun h1 : p => h1)
    (fun h1 : ¬p => absurd h1 h)

example (h : ¬¬p) : p :=
  byContradiction h

example (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
  byCases
    (fun hp : p =>
      Or.inr (fun hq : q => h ⟨hp, hq⟩))
    (fun hnp : ¬p => Or.inl hnp)

end cl

-- Exercises
section ex_const
variable (p q r : Prop)

-- commutativity of ∧ and ∨
example : p ∧ q ↔ q ∧ p := {
  mp := fun (hpq : p ∧ q) => {
    left := hpq.right,
    right := hpq.left,
  }
  mpr := fun (hqp : q ∧ p) => {
    left := hqp.right,
    right := hqp.left,
  }
}

example : p ∨ q ↔ q ∨ p := {
  mp := fun (hpq : p ∨ q) =>
    match hpq with
      | Or.inl hp => Or.inr hp
      | Or.inr hq => Or.inl hq
  mpr := fun (hqp : q ∨ p) =>
    match hqp with
      | Or.inl hq => Or.inr hq
      | Or.inr hp => Or.inl hp
}

-- associativity of ∧ and ∨
example : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r) := {
  mp := fun (hpqr : (p ∧ q) ∧ r) => {
    left := hpqr.left.left,
    right :=  {
      left := hpqr.left.right,
      right := hpqr.right,
    },
  },
  mpr := fun (hpqr : p ∧ (q ∧ r)) => {
    left :=  {
      left := hpqr.left,
      right := hpqr.right.left,
    },
    right := hpqr.right.right,
  },
}

example : (p ∨ q) ∨ r ↔ p ∨ (q ∨ r) := {
  mp := fun (hprq : (p ∨ q) ∨ r) =>
    match hprq with
      | Or.inl hpq =>
        match hpq with
          | Or.inl hp => Or.inl hp
          | Or.inr hq => Or.inr (Or.inl hq)
      | Or.inr hr => Or.inr (Or.inr hr),
  mpr := fun (hprq : p ∨ (q ∨ r)) =>
    match hprq with
      | Or.inl hp => Or.inl (Or.inl hp)
      | Or.inr hqr =>
        match hqr with
          | Or.inl hq => Or.inl (Or.inr hq)
          | Or.inr hr => Or.inr hr,
}

-- distributivity
example : p ∧ (q ∨ r) ↔ (p ∧ q) ∨ (p ∧ r) := {
  mp := fun (hpqr : p ∧ (q ∨ r)) =>
    match hpqr.right with
      | Or.inl hq => Or.inl {
        left := hpqr.left,
        right := hq,
      }
      | Or.inr hr => Or.inr {
        left := hpqr.left,
        right := hr,
      },
  mpr := fun (hpqpr : (p ∧ q) ∨ (p ∧ r)) =>
    match hpqpr with
      | Or.inl hpq => {
        left := hpq.left,
        right := Or.inl hpq.right,
      }
      | Or.inr hpr => {
        left := hpr.left,
        right := Or.inr hpr.right,
      },
}

example : p ∨ (q ∧ r) ↔ (p ∨ q) ∧ (p ∨ r) := {
  mp := fun (hpqr : p ∨ (q ∧ r)) => {
    left := match hpqr with
      | Or.inl hp => Or.inl hp
      | Or.inr hqr => Or.inr hqr.left,
    right := match hpqr with
      | Or.inl hp => Or.inl hp
      | Or.inr hqr => Or.inr hqr.right,
  },
  mpr := fun (hpqpr : (p ∨ q) ∧ (p ∨ r)) =>
    match hpqpr.left, hpqpr.right with
      | Or.inl hp, _ => Or.inl hp
      | _, Or.inl hp => Or.inl hp
      | Or.inr hq, Or.inr hr => Or.inr {
        left := hq, right := hr,
      },
}

-- other properties
example : (p → (q → r)) ↔ (p ∧ q → r) := {
  mp := fun (fcr : p → (q → r)) =>
    fun (hpq : p ∧ q) => fcr hpq.left hpq.right,
  mpr := fun (f : p ∧ q → r) =>
    fun (hp : p) (hq : q) => f { left := hp, right := hq },
}

example : ((p ∨ q) → r) ↔ (p → r) ∧ (q → r) := sorry
example : ¬(p ∨ q) ↔ ¬p ∧ ¬q := sorry
example : ¬p ∨ ¬q → ¬(p ∧ q) := sorry
example : ¬(p ∧ ¬p) := sorry
example : p ∧ ¬q → ¬(p → q) := sorry
example : ¬p → (p → q) := sorry
example : (¬p ∨ q) → (p → q) := sorry
example : p ∨ False ↔ p := sorry
example : p ∧ False ↔ False := sorry
example : (p → q) → (¬q → ¬p) := sorry

end ex_const

section ex_cls
variable (p q r : Prop)

example : (p → q ∨ r) → ((p → q) ∨ (p → r)) := sorry
example : ¬(p ∧ q) → ¬p ∨ ¬q := sorry
example : ¬(p → q) → p ∧ ¬q := sorry
example : (p → q) → (¬p ∨ q) := sorry
example : (¬q → ¬p) → (p → q) := sorry
example : p ∨ ¬p := sorry
example : (((p → q) → p) → p) := sorry

end ex_cls
