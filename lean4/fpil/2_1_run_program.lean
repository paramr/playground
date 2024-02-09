-- Running a Program
-- $ lean --run 2_1_run_program.lean

def codeblock1 := IO.println "Hello World!"

-- Combining IO Actions
def codeblock2 : IO Unit := do
  let stdin <- IO.getStdin
  let stdout <- IO.getStdout
  stdout.putStrLn "How would you like to be addressed?"
  let gl := stdin.getLine
  let input <- gl
  let name := input.dropRightWhile Char.isWhitespace
  stdout.putStrLn s!"Hello, {name}!"

-- Runner
def main : IO Unit :=
  codeblock2
