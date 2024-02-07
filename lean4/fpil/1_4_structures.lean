-- Structures
#check 1.2
#check -454.2123215
#check 0.0
#check 0
#check (0 : Float)

structure Point where
  x : Float
  y : Float
  deriving Repr

def origin : Point := { x := 0.0, y := 0.0 }

#eval origin
#eval origin.x
#eval origin.y

def addPoints (p1 : Point) (p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

#eval addPoints { x := 1.5, y := 32 } { x := -8, y := 0.2 }

def distance (p1 : Point) (p2 : Point) : Float :=
  Float.sqrt (((p2.x - p1.x) ^ 2.0) + ((p2.y - p1.y) ^ 2.0))

#eval distance { x := 1.0, y := 2.0 } { x := 5.0, y := -1.0 }

-- Error
-- #check { x := 0.0, y := 0.0 }
#check ({ x := 0.0, y := 0.0 } : Point)

structure Point3D where
  x : Float
  y : Float
  z : Float
  deriving Repr

def origin3D : Point3D := { x := 0.0, y := 0.0, z := 0.0 }

-- Updating Structures
def zeroX_not_good (p : Point) : Point :=
  { x := 0, y := p.y }

def zeroX (p : Point) : Point :=
  { p with x := 0 }

def fourAndThree : Point :=
  { x := 4.3, y := 3.4 }

#eval fourAndThree
#eval zeroX fourAndThree
#eval fourAndThree

-- Behind the Scenes
#check Point.mk 1.5 2.8
#check Point.mk
#check (Point.mk)

structure XPoint where
  point ::
  x : Float
  y : Float
  deriving Repr

#check XPoint.point
#check (XPoint.point)


#check (Point.x)
#check (Point.y)

#eval Point.x origin

#eval "one string".append " and another"

-- Why does it allow arbitrary lcoation for p the Point?
def Point.modifyBoth (f : Float -> Float) (p : Point) : Point :=
  { x := f p.x, y := f p.y }

#eval fourAndThree.modifyBoth Float.floor

-- Exercies
structure RectangularPrism where
  height : Float
  width : Float
  depth : Float
  deriving Repr

def volume (prsm : RectangularPrism) : Float :=
  prsm.depth * prsm.height * prsm.width

structure Segment where
  p1 : Point
  p2 : Point
  deriving Repr

def length (ls : Segment) :=
  Float.sqrt (ls.p1.x - ls.p2.x) ^ 2 + (ls.p1.y - ls.p2.y) ^ 2

-- RectangularPrism.mk, RectangularPrism.height,
-- RectangularPrism.width, RectangularPrism.depth

-- structure Hamster where
--   name : String
--   fluffy : Bool
-- Hamster.mk, Hamster.name, Hamster.fluffy

-- structure Book where
--   makeBook ::
--   title : String
--   author : String
--   price : Float
-- Book.makeBook, Book.title, Book.author, Book.price
