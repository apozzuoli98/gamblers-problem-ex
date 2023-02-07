# gamblers-problem-ex
Implementation of Gambler's Problem example from Sutton &amp; Barto "Reinforcement Learning: An Introduction," 2nd Ed. 2020

A gambler makes bets on a sequence of coin flips. If heads, the gambler wins what they staked. If tails, they lose the stake.
The game ends when the gambler reaches $100 or $0.
Ph is a variable denoting the probability of heads. The default value in this example is ph = 0.4.


The file "gamblersProblem.py" runs with the command:

python3 gamblersProblem.py


In the console, the final output is the value of each state in an array, followed by the policy for each state in an array.
