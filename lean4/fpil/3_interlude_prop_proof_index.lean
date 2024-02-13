-- Interlude: Propositions, Proofs, and Indexing

def woodlandCritters : List String :=
  ["hedgehog", "deer", "snail"]

def hedgehog := woodlandCritters[0]
def deer := woodlandCritters[1]
def snail := woodlandCritters[2]

#eval woodlandCritters[3]!
#eval woodlandCritters[3]?

-- Propositions and Proofs
def onePlusOneIsTwo : 1 + 1 = 2 := rfl
def onePlusOneIsThree : 1 + 1 = 3 := rfl

def OnePlusOneIsTwo : Prop := 1 + 1 = 2
theorem onePlusOneIsTwoTh1 : OnePlusOneIsTwo := rfl
theorem onePlusOneIsTwoTh2 : 1 + 1 = 2 := rfl

-- Tactics
theorem onePlusOneIsTwoTac : 1 + 1 = 2 := by
  simp

-- Connectives
theorem addAndAppend : 1 + 1 = 2 ∧ "Str".append "ing" = "String" :=
  And.intro rfl rfl
theorem addAndAppendTac : 1 + 1 = 2 ∧ "Str".append "ing" = "String" :=
  by trivial

theorem andImpliesOr : A ∧ B -> A ∨ B :=
  fun andEvidence =>
    match andEvidence with
      | And.intro a _ => Or.inl a

theorem onePlusOneOrLessThan : 1 + 1 = 2 ∨ 3 < 5 := by simp
theorem notTwoEqualFive : ¬(1 + 1 = 5) := by simp
theorem trueIsTrue : True := True.intro
theorem trueOrFalse : True ∨ False := by simp
theorem falseImpliesTrue : False -> True := by simp

-- Evidence as Arguments
def thirdBad (xs : List T) : T := xs[2]
def thirdByEvidence (xs : List T) (ok : xs.length > 2) : T := xs[2]

#eval thirdByEvidence woodlandCritters (by trivial)

-- Indexing Without Evidence
def thirdOption (xs : List T) : Option T := xs[2]?

#eval thirdOption woodlandCritters
#eval thirdOption ["only", "two"]

-- because T can be empty
def unsafeThird (xs : List T) : T := xs[2]!

-- Exercises
theorem twoPlusThreeIsFive : 2 + 3 = 5 := rfl
theorem fifteenMinusEightIsSeven : 15 - 8 = 7 := rfl
theorem appendHelloAndWorld : "Hello, ".append "world" = "Hello, world" := rfl
-- rfl is applicableony for equality
theorem fiveLessthanEighteen : 5 < 18 := rfl

theorem twoPlusThreeIsFiveBySimp : 2 + 3 = 5 := by simp
theorem fifteenMinusEightIsSevenBySimp : 15 - 8 = 7 := by simp
theorem appendHelloAndWorldByTrivial : "Hello, ".append "world" = "Hello, world" := by trivial
-- rfl is applicableony for equality
theorem fiveLessthanEighteenByTrivial : 5 < 18 := by trivial

def fifthByEvidence (xs : List T) (ok : xs.length > 4): T := xs[4]
