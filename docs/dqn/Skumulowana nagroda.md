$$
G_t=R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+...=\sum_{k=0}^{\infty}{\gamma^kR_{t+k+1}} \tag{1}
$$
$\gamma \in [0, 1]$ jest współczynnikiem zmniejszającym, nagrodę w czasie. Za pomocą tego współczynnika można sterować tym, czy agent ma się kierować perspektywą krótko-, czy długofalową. Przy niskiej wartości $\gamma$, znaczenie kolejnych nagród zanika, co wspiera podejmowanie decyzji, których oczekiwana wartość jest uzależniona od najbliższych nagród.