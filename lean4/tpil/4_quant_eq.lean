-- Quantifiers and Equality

-- The Universal Quantifier
section uni_quant

example (α : Type) (p q : α → Prop) : (∀ x : α, p x ∧ q x) → ∀ x : α, p x :=
  fun h : ∀ x : α, p x ∧ q x =>
    fun x : α =>
      (h x).left

variable (α : Type) (rel : α → α → Prop)
variable (trans_re : ∀ x y z, rel x y → rel y z → rel x z)
variable (trans_r : ∀ {x y z}, rel x y → rel y z → rel x z)

variable (a b c : α)
variable (hab : rel a b) (hbc : rel b c)

#check hab
#check sorryAx

#check trans_re
#check trans_re a b c
#check trans_re a b c hab
#check trans_re a b c hab hbc

#check trans_r
#check trans_r hab
#check trans_r hab hbc

variable (refl_r : ∀ x, rel x x)
variable (symm_r : ∀ {x y}, rel x y → rel y x)

example (a b c d : α) (hab : rel a b) (hcb : rel c b) (hcd : rel c d) : rel a d :=
  trans_r hab (trans_r (symm_r hcb) hcd)

end uni_quant

-- Equality
section eq

#check Eq.refl
#check Eq.symm
#check Eq.trans

universe u
#check @Eq.refl.{u}
#check @Eq.symm.{u}
#check @Eq.trans.{u}

variable (α β : Type) (a b c d : α)
variable (hab : a = b) (hcb : c = b) (hcd : c = d)

example : a = d := Eq.trans (Eq.trans hab (Eq.symm hcb)) hcd
example : a = d := (hab.trans hcb.symm).trans hcd

example (a : α) (b : β) : (a, b).1 = a := Eq.refl a
example (f : α → β) (a : α) : (fun x => f x) a = f a := Eq.refl _
example : 2 + 3 = 5 := rfl

example
    (α : Type)
    (a b : α)
    (p : α → Prop)
    (h1 : a = b)
    (h2 : p a) : p b :=
  Eq.subst h1 h2

variable (f g : α → Nat)
variable (h1 : a = b)
variable (h2 : f = g)

example : f a = f b := congrArg f h1
example : f a = g a := congrFun h2 a
example : f a = g b := congr h2 h1

variable (l m n : Nat)

example : n + 0 = n := Nat.add_zero n
example : 0 + n = n := Nat.zero_add n
example : n * 1 = n := Nat.mul_one n
example : 1 * n = n := Nat.one_mul n
example : n + m = m + n := Nat.add_comm n m
example : n + m + l = n + (m + l) := Nat.add_assoc n m l
example : n * m = m * n := Nat.mul_comm n m
example : n * m * l = n * (m * l) := Nat.mul_assoc n m l
example : n * (m + l) = n * m + n * l := Nat.left_distrib n m l
example : (n + m) * l = n * l + m * l := Nat.right_distrib n m l

example (x y : Nat) : (x + y) * (x + y) = x * x + y * x + x * y + y * y :=
  have h1 : (x + y) * (x + y) = (x + y) * x + (x + y) * y :=
    Nat.mul_add (x + y) x y
  have h2 : (x + y) * (x + y) = x * x + y * x + (x * y + y * y) :=
    (Nat.add_mul x y x) ▸ (Nat.add_mul x y y) ▸ h1
  h2.trans (Nat.add_assoc (x * x + y * x) (x * y) (y * y)).symm

end eq

-- Calculational Proofs
section calc_prf
variable (a b c d e : Nat)
variable (h1 : a = b)
variable (h2 : b = c + 1)
variable (h3 : c = d)
variable (h4 : e = 1 + d)

theorem T0 : a = e :=
  calc
    a = b := h1
    _ = c + 1 := h2
    _ = d + 1 := congrArg Nat.succ h3
    _ = 1 + d := Nat.add_comm d 1
    _ = e := Eq.symm h4

theorem T1 : a = e :=
  calc
    a = b      := by rw [h1]
    _ = c + 1  := by rw [h2]
    _ = d + 1  := by rw [h3]
    _ = 1 + d  := by rw [Nat.add_comm]
    _ = e      := by rw [h4]

theorem T2 : a = e :=
  calc
    a = d + 1  := by rw [h1, h2, h3]
    _ = 1 + d  := by rw [Nat.add_comm]
    _ = e      := by rw [h4]

theorem T3 : a = e :=
  by rw [h1, h2, h3, Nat.add_comm, h4]

theorem T4 : a = e :=
  by simp [h1, h2, h3, Nat.add_comm, h4]

example
    (a b c d : Nat)
    (h1 : a = b)
    (h2 : b ≤ c)
    (h3 : c + 1 < d)
    : a < d :=
  calc
    a = b     := h1
    _ < b + 1 := Nat.lt_succ_self b
    _ ≤ c + 1 := Nat.succ_le_succ h2
    _ < d     := h3

def divides (x y : Nat) : Prop :=
  ∃ k, k*x = y

def divides_trans (h1 : divides x y) (h2 : divides y z) : divides x z :=
  let ⟨k1, d1⟩ := h1
  let ⟨k2, d2⟩ := h2
  ⟨k1 * k2, by rw [Nat.mul_comm k1 k2, Nat.mul_assoc, d1, d2]⟩

def divides_mul (x : Nat) (k : Nat) : divides x (k*x) :=
  ⟨k, rfl⟩

instance : Trans divides divides divides where
  trans := divides_trans

example (h1 : divides x y) (h2 : y = z) : divides x (2*z) :=
  calc
    divides x y     := h1
    _ = z           := h2
    divides _ (2*z) := divides_mul ..

example (x y : Nat) : (x + y) * (x + y) = x * x + x * y + x * y + y * y :=
  calc
    (x + y) * (x + y) = (x + y) * x + (x + y) * y  := by rw [Nat.mul_add]
    _ = x * x + y * x + (x + y) * y                := by rw [Nat.add_mul]
    _ = x * x + y * x + (x * y + y * y)            := by rw [Nat.add_mul]
    _ = x * x + y * x + x * y + y * y              := by rw [←Nat.add_assoc]
    _ = x * x + x * y + x * y + y * y              := by simp [Nat.mul_comm]

example (x y : Nat) : (x + y) * (x + y) = x * x + x * y + x * y + y * y :=
  calc (x + y) * (x + y)
    _ = (x + y) * x + (x + y) * y       := by rw [Nat.mul_add]
    _ = x * x + y * x + (x + y) * y     := by rw [Nat.add_mul]
    _ = x * x + y * x + (x * y + y * y) := by rw [Nat.add_mul]
    _ = x * x + y * x + x * y + y * y   := by rw [←Nat.add_assoc]
    _ = x * x + x * y + x * y + y * y   := by simp [Nat.mul_comm]

example (x y : Nat) : (x + y) * (x + y) = x * x + y * x + x * y + y * y :=
  by rw [Nat.mul_add, Nat.add_mul, Nat.add_mul, ←Nat.add_assoc]

example (x y : Nat) : (x + y) * (x + y) = x * x + x * y + x * y + y * y :=
  by simp [Nat.mul_add, Nat.add_mul, Nat.add_assoc, Nat.mul_comm]

end calc_prf

-- The Existential Quantifier
section ex_quant

example : ∃ x : Nat, x > 0 :=
  have h : 1 > 0 := Nat.zero_lt_succ 0
  Exists.intro 1 h

example (x : Nat) (h : x > 0) : ∃ y, y < x :=
  ⟨0, h⟩

example (x y z : Nat) (hxy : x < y) (hyz : y < z) : ∃ w, x < w ∧ w < z :=
  ⟨y, hxy, hyz⟩

#check @Exists.intro

variable (g : Nat → Nat → Nat)
variable (hg : g 0 0 = 0)
variable (α : Type) (p q : α → Prop)
variable (a : α)
variable (r : Prop)

theorem gex1 : ∃ x, g x x = x := ⟨0, hg⟩
theorem gex2 : ∃ x, g x 0 = x := ⟨0, hg⟩
theorem gex3 : ∃ x, g 0 0 = x := ⟨0, hg⟩
theorem gex4 : ∃ x, g x x = 0 := ⟨0, hg⟩

set_option pp.explicit true  -- display implicit arguments
#print gex1
#print gex2
#print gex3
#print gex4

example (h : ∃ x, p x ∧ q x) : ∃ x, q x ∧ p x :=
  Exists.elim h
    (fun (w) (hw : p w ∧ q w) =>
     show ∃ x, q x ∧ p x from ⟨w, hw.right, hw.left⟩)

example (h : ∃ x, p x ∧ q x) : ∃ x, q x ∧ p x :=
  match h with
  | ⟨w, hw⟩ => ⟨w, hw.right, hw.left⟩

example (h : ∃ x, p x ∧ q x) : ∃ x, q x ∧ p x :=
  match h with
  | ⟨(w : α), (hw : p w ∧ q w)⟩ => ⟨w, hw.right, hw.left⟩

example (h : ∃ x, p x ∧ q x) : ∃ x, q x ∧ p x :=
  match h with
  | ⟨w, hpw, hqw⟩ => ⟨w, hqw, hpw⟩

example (h : ∃ x, p x ∧ q x) : ∃ x, q x ∧ p x :=
  let ⟨w, hpw, hqw⟩ := h
  ⟨w, hqw, hpw⟩

example : (∃ x, p x ∧ q x) → ∃ x, q x ∧ p x :=
  fun ⟨w, hpw, hqw⟩ => ⟨w, hqw, hpw⟩

def is_even (a : Nat) := ∃ b, a = 2 * b

theorem even_plus_even_0 (h1 : is_even n) (h2 : is_even m) : is_even (n + m) :=
  Exists.elim h1 (fun w1 (hw1 : n = 2 * w1) =>
  Exists.elim h2 (fun w2 (hw2 : m = 2 * w2) =>
    Exists.intro (w1 + w2)
      (calc n + m
        _ = 2 * w1 + 2 * w2 := by rw [hw1, hw2]
        _ = 2 * (w1 + w2)   := by rw [Nat.mul_add])))

theorem even_plus_even_1 (h1 : is_even n) (h2 : is_even m) : is_even (n + m) :=
  match h1, h2 with
  | ⟨w1, hw1⟩, ⟨w2, hw2⟩ => ⟨w1 + w2, by rw [hw1, hw2, Nat.mul_add]⟩

section classical
open Classical

example (h : ¬ ∀ x, ¬ p x) : ∃ x, p x :=
  byContradiction
    fun h1 : ¬ ∃ x, p x =>
      have h2 : ∀ x, ¬ p x :=
        fun x (h3 : p x) =>
          have h4 : ∃ x, p x := ⟨x, h3⟩
          show False from h1 h4
      show False from h h2

example : (∃ x, p x ∨ q x) ↔ (∃ x, p x) ∨ (∃ x, q x) := {
  mp := fun ⟨a, (h1 : p a ∨ q a)⟩ =>
      Or.elim h1
        (fun hpa : p a => Or.inl ⟨a, hpa⟩)
        (fun hqa : q a => Or.inr ⟨a, hqa⟩),
  mpr := fun h : (∃ x, p x) ∨ (∃ x, q x) =>
      Or.elim h
        (fun ⟨a, hpa⟩ => ⟨a, (Or.inl hpa)⟩)
        (fun ⟨a, hqa⟩ => ⟨a, (Or.inr hqa)⟩),
}

example : (∃ x, p x → r) ↔ (∀ x, p x) → r := {
  mp := fun ⟨b, (hb : p b → r)⟩ =>
     fun h2 : ∀ x, p x =>
     show r from hb (h2 b)
  mpr := fun h1 : (∀ x, p x) → r =>
     show ∃ x, p x → r from
       byCases
         (fun hap : ∀ x, p x => ⟨a, λ _ => h1 hap⟩)
         (fun hnap : ¬ ∀ x, p x =>
          byContradiction
            (fun hnex : ¬ ∃ x, p x → r =>
              have hap : ∀ x, p x :=
                fun x =>
                byContradiction
                  (fun hnp : ¬ p x =>
                    have hex : ∃ x, p x → r := ⟨x, (fun hp => absurd hp hnp)⟩
                    show False from hnex hex)
              show False from hnap hap))
}

end classical

end ex_quant

-- More on the Proof Language
section more_prf_lang
variable (f : Nat → Nat)
variable (h : ∀ x : Nat, f x ≤ f (x + 1))

example : f 0 ≤ f 3 :=
  have : f 0 ≤ f 1 := h 0
  have : f 0 ≤ f 2 := Nat.le_trans this (h 1)
  show f 0 ≤ f 3 from Nat.le_trans this (h 2)

example : f 0 ≤ f 3 :=
  have : f 0 ≤ f 1 := h 0
  have : f 0 ≤ f 2 := Nat.le_trans (by assumption) (h 1)
  show f 0 ≤ f 3 from Nat.le_trans (by assumption) (h 2)

example : f 0 ≥ f 1 → f 1 ≥ f 2 → f 0 = f 2 :=
  fun _ : f 0 ≥ f 1 =>
  fun _ : f 1 ≥ f 2 =>
  have : f 0 ≥ f 2 := Nat.le_trans ‹f 1 ≥ f 2› ‹f 0 ≥ f 1›
  have : f 0 ≤ f 2 := Nat.le_trans (h 0) (h 1)
  show f 0 = f 2 from Nat.le_antisymm this ‹f 0 ≥ f 2›

end more_prf_lang

-- Exercises
section exercises
variable (α : Type) (p q : α → Prop)
variable (r : Prop)

example : (∀ x, p x ∧ q x) ↔ (∀ x, p x) ∧ (∀ x, q x) := sorry
example : (∀ x, p x → q x) → (∀ x, p x) → (∀ x, q x) := sorry
example : (∀ x, p x) ∨ (∀ x, q x) → ∀ x, p x ∨ q x := sorry

example : α → ((∀ x : α, r) ↔ r) := sorry
example : (∀ x, p x ∨ r) ↔ (∀ x, p x) ∨ r := sorry
example : (∀ x, r → p x) ↔ (r → ∀ x, p x) := sorry

variable (men : Type) (barber : men)
variable (shaves : men → men → Prop)

example (h : ∀ x : men, shaves barber x ↔ ¬ shaves x x) : False := sorry

def even (n : Nat) : Prop := sorry

def prime (n : Nat) : Prop := sorry

def infinitely_many_primes : Prop := sorry

def Fermat_prime (n : Nat) : Prop := sorry

def infinitely_many_Fermat_primes : Prop := sorry

def goldbach_conjecture : Prop := sorry

def Goldbach's_weak_conjecture : Prop := sorry

def Fermat's_last_theorem : Prop := sorry

end exercises

section exercises_classical

open Classical

variable (α : Type) (p q : α → Prop)
variable (r : Prop)

example : (∃ x : α, r) → r := sorry
example (a : α) : r → (∃ x : α, r) := sorry
example : (∃ x, p x ∧ r) ↔ (∃ x, p x) ∧ r := sorry
example : (∃ x, p x ∨ q x) ↔ (∃ x, p x) ∨ (∃ x, q x) := sorry

example : (∀ x, p x) ↔ ¬ (∃ x, ¬ p x) := sorry
example : (∃ x, p x) ↔ ¬ (∀ x, ¬ p x) := sorry
example : (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := sorry
example : (¬ ∀ x, p x) ↔ (∃ x, ¬ p x) := sorry

example : (∀ x, p x → r) ↔ (∃ x, p x) → r := sorry
example (a : α) : (∃ x, p x → r) ↔ (∀ x, p x) → r := sorry
example (a : α) : (∃ x, r → p x) ↔ (r → ∃ x, p x) := sorry

end exercises_classical
