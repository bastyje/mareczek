Używając tych metod uczymy się [[Funkcja wartości stanu|funkcji wartości stanu]]. Nie uczymy się polityki $\pi$, natomiast należy pamiętać o tym jaki jest [[Cel agenta RL|cel agenta RL]]. W związku z tym, w tej klasie metod również będziemy potrzebować polityki. Jednak w przypadku metod opartych o wartość, polityka jest jedynie predefiniowaną funkcją, która operuje w oparciu o zwracają wartość przez funkcję wartości.
$$
\pi^*(s)=\arg \max_a Q^*(s, a) \tag{1}
$$
Równanie $(1)$ pokazuje, że znalezienie optymalnej funkcji wartości równoważne jest znalezieniu optymalnej polityki ($Q$ oznacza [[Funkcja wartości akcji|funkcję wartości akcji]]).

Wyróżnia się dwa rodzaje funkcji wartości:
- [[Funkcja wartości stanu|Funkcje wartości stanu]]
- [[Funkcja wartości stanu-akcji|Funkcje wartości stanu-akcji]]