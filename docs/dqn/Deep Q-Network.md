Jest to sieć neuronowa, wykorzystująca [[Q-Learning]], która jako wejście przyjmuje macierz stanu (lub tensor kilku stanów następujących po sobie), a na wyjściu podaje wektor Q-wartości wszystkich możliwych do podjęcia akcji.

## Architektura
![[dqn-arch.svg]]
## Algorytm
- $D$ $-$ bufor doświadczeń, w którym zapisywana jest krotka $(s, a, r, s')$
- $Q'$ $-$ Q-funkcja celu
- $Q$ $-$ Q-funkcja
- $s_x$ $-$ sekwencja stanów
- $\phi(s_x)$ $-$ funkcja preprocesująca sekwencję

[[Polityka epsilon-zachłanna]] dla DQN wygląda następująco:
Z prawdopodobieństwem $\epsilon$ wybierz losową akcję
Z prawdopodobieństwem $1-\epsilon$ wybierz akcję: $$
a_t = \arg \max_a Q(\phi(s_t), a, \theta ) \tag{1}
$$
1. Zainicjuj bufor doświadczeń $D$
2. Zainicjuj $Q$ losowymi wagami $\theta$
3. Zainicjuj $Q'$ losowymi wagami $\theta' = \theta$
4. Dla każdego epizodu z $M$ epizodów:
	1. Zainicjuj sekwencję $s_1 = {x_1}$ i preprocesowaną sekwencję $\phi_1 = \phi(s_1)$
	2. Dla każdego kroku czasowego $t \in T$
		1. Wybierz akcję $a_t$ zgodnie z równaniem $(1)$
		2. Wykonaj akcję i obserwuj $r_t$ oraz obraz $x_{t+1}$
		3. $s_{t+1} \leftarrow s_t, a_t, x_{t+1}$
		4. $\phi_{t+1} \leftarrow \phi(s_{t+1})$
		5. Zapisz tranzycję $(\phi_t, a_t, r_t, \phi_{t+1})$ w $D$
		6. Wybierz mini-serię (minibatch) tranzycji $(\phi_j, a_j, r_j, \phi_{j+1})$ z $D$
		7. $y_j \leftarrow r_j\ jeśli\ epizod\ kończy\ się\ w\ j+1,\ inaczej\ r_j + \gamma \max_{a'}Q'(\phi_{j+1}, a', \theta')$
		8. Przeprowadź spadek gradientu na $(y_j - Q(\phi_j, a_j, \theta))^2$ w stosunku do parametrów $\theta$
		9. Co $C$ kroków wykonaj $Q' \leftarrow Q$
		
W tym algorytmie wykorzystane są takie techniki, jak [[Ponawianie doświadczeń|ponawianie doświadczeń]], czy [[Sieć celu|sieć celu]].