Polega ono na tym, że doświadczenia $(s_t, a_t, r_t, s_{t+1})$ nabyte podczas interakcji ze środowiskiem są zapisywane i trzymane w buforze doświadczeń $D$.

Ponawianie doświadczeń spełnia dwie funkcje:
1. Bardziej efektywne korzystanie z doświadczeń podczas treningu
2. Redukcja problemu katastroficznego zapominania

Dzięki temu, że próbki są losowo wybierane z buforu doświadczeń, unika się korelacji pomiędzy sekwencjami operacji.
