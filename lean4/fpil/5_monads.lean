-- Monads

-- Checking for none: Don't Repeat Yourself
def firstThirdFifth (xs : List T) : Option (T × T × T) :=
  match xs[0]? with
  | none => none
  | some first =>
    match xs[2]? with
    | none => none
    | some third =>
      match xs[4]? with
      | none => none
      | some fifth =>
        some (first, third, fifth)

#eval firstThirdFifth [1, 2, 3, 4, 5, 6, 7]

def andThen (opt : Option T) (next : T → Option U) : Option U :=
  match opt with
  | none => none
  | some x => next x

def firstThirdFifthMon (xs : List T) : Option (T × T × T) :=
  andThen xs[0]? fun first =>
    andThen xs[2]? fun third =>
      andThen xs[4]? fun fifth =>
        some (first, third, fifth)

#eval firstThirdFifthMon [1, 2, 3, 4, 5, 6, 7]
