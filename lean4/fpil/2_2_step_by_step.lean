-- Running a Program
-- $ lean --run 2_2_step_by_step.lean

-- IO Actions as Values
def twice (action : IO Unit) : IO Unit := do
  action
  action

def nTimes (action : IO Unit) : Nat -> IO Unit
  | 0 => pure ()
  | n + 1 => do
    action
    nTimes action n

def codeblock1 : IO Unit := do
  let action := IO.println "Shy"
  twice action
  IO.println "---"
  nTimes action 3

def countdown : Nat -> List (IO Unit)
  | 0 => [IO.println "Blast Off!!!"]
  | n + 1 => IO.println s!"{n + 1}" :: countdown n

def from5 : List (IO Unit) := countdown 5
-- #eval from5.length

def runActions : List (IO Unit) -> IO Unit
  | [] => pure ()
  | act :: actions => do
    act
    runActions actions

def codeblock2 : IO Unit := do
  runActions (countdown 10)

-- Exercise
def codeblock3 : IO Unit := do
  let englishGreeting := IO.println "Hello!"
  IO.println "Bonjour!"
  englishGreeting

-- Runner
def main : IO Unit :=
  codeblock3
