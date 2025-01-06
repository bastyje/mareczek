[[Off-policy]] metoda RL, w której wykorzystuje się [[Uczenie metodą różnic czasowych|uczenie metodą różnic czasowych]] do wytrenowania [[Funkcja wartości stanu-akcji|funkcji wartości stanu-akcji (Q-funkcji)]]. Q-funkcja w rzeczywistości jest tabelą, w której wierszach mamy wszystkie możliwe stany środowiska, a w kolumnach akcje, które są możliwe do podjęcia. Na przecięciu stanu i akcji znajduje się wartość Q-funkcji, czyli [[Skumulowana nagroda|skumulowana nagroda]] dla danego stanu.

## Algorytm
1. Zainicjuj Q-tablicę dowolnie (zerami, losowo, itp.). Stanom terminalnym należy przypisać wartość 0.
2. Dla każdego epizodu:
	1. Wybierz akcję $a$ w aktualnym stanie $s$ zgodnie ze stosowaną polityką (np. [[Polityka epsilon-zachłanna|epsilon-zachłanna]])
	2. Podejmij akcję i obserwuj wartości $R_{t+1}$ oraz $S_{t+1}$
	3. Zaktualizuj Q-tablicę zgodnie z [[Uczenie metodą różnic czasowych|metodą różnic czasowych]]
	4. Kroki 1-3 powtarzaj dopóki $S_t$ jest stanem nieterminalnym
