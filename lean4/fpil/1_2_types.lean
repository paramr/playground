-- Types
#eval (1 + 2 : Nat)

-- Why is this not a type error. This is def not 0.
#eval 1 - 2

#eval (1 - 2 : Int)

#check (1 - 2 : Int)

#check 1 - 2

-- Errors
-- #check String.append "hello" [" ", "world"]
