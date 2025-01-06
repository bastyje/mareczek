$$
V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)] \tag{1}
$$
$$
Q(S_t, A_t)\leftarrow Q(S_t, A_t)+\alpha [R_{t+1}+\gamma \max_a Q(S_{t+1}, a)-Q(S_t, A_t)] \tag{2}
$$
W odróżnieniu od [[Uczenie metodą Monte Carlo|metody Monte Carlo]], w podejściu różnic skończonych czekamy nie na ukończenie epizodu a na kolejny stan. Ponieważ nie jest znana skumulowana nagroda za cały epizod, używa się podejścia zaczerpniętego z [[Równanie Bellmana|równania Bellmana]], czyli dodania do aktualnej nagrody przecenionej skumulowanej nagrody. Równanie $(1)$ jest dla [[Funkcja wartości stanu|funkcji wartości stanu]], natomiast równanie $(2)$ jest dla [[Funkcja wartości stanu-akcji|funkcji wartości stanu-akcji]].