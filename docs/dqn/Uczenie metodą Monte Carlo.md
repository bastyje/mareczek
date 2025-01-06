Metoda Monte Carlo zakłada przejście całego epizodu interakcji przed aktualizacją funkcji wartości. Sposób aktualizacji pokazany jest na równaniu $(1)$. Do aktualnej funkcji $V(S_t)$ dodawana jest różnica pomiędzy skumulowaną nagrodą $G_t$ a aktualną estymacją $V(S_t)$ pomnożona o współczynnik uczenia $\alpha$. 
$$
V(S_t) \leftarrow V(S_t) + \alpha[G_t-V(S_t)] \tag{1}
$$
