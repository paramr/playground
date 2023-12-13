{-# OPTIONS --guardedness #-}

module prog where

open import IO

main : Main
main = run (putStrLn "Hello, World!")