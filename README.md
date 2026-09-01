# Estudio e implementación paralela de métodos explícitos multipaso
Estudio e implementación paralela de métodos explícitos multipaso, Resolución de Ecuaciones de Advección-Reacción-Difusión

En este Trabajo de Fin de Grado se realiza una investigación en el mundo de las ecuaciones diferenciales ordinarias y los métodos numéricos que se utilizan para alcanzar soluciones aproximadas de las mismas a través de la computación.

Para este fin, se estudian tres métodos de resolución distintos, Runge-Kutta, Adams-Bashforth, y Adams-Bashforth-Moulton, y se implementan. A partir de este punto, se plantean dos tecnologías diferentes que permiten paralelizar el código en busca de una ganancia en tiempo. 

La primera de éstas es OpenMP, que trabaja el paralelismo en CPUs multinúcleo, en el cual se implementan y se comparan dos estrategias de paralelismo opuestas. Y la segunda es CUDA, que aprovecha el cómputo masivo y paralelo de las tarjetas gráficas modernas. 

Partiendo de una serie de problemas secuenciales, se implementan y paralelizan con ambas tecnologías, y se realizan mediciones en el tiempo.
