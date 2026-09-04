# Estudio e implementación paralela de métodos explícitos multipaso
Estudio e implementación paralela de métodos explícitos multipaso para la resolución de Ecuaciones de Advección-Reacción-Difusión

En este Trabajo de Fin de Grado se realiza una investigación en el mundo de las ecuaciones diferenciales ordinarias y los métodos numéricos que se utilizan para alcanzar soluciones aproximadas a las mismas a través de la computación.

Para este fin, se estudian tres familias de métodos de resolución distintos, Runge-Kutta, Adams-Bashforth, y Adams-Bashforth-Moulton, y se implementan. A partir de este punto, se plantean dos tecnologías diferentes que permiten paralelizar el código en busca de una ganancia en tiempo. 

La primera de éstas es OpenMP, que explota el paralelismo sobre CPUs multinúcleo, en el cual se implementan y se comparan dos estrategias de paralelización complementarias. Y la segunda es CUDA, que aprovecha el paralelismo masivo de las tarjetas gráficas modernas de NVIDIA. 

Partiendo de una serie de algoritmos secuenciales y modelos matemáticos, se implementan y paralelizan con ambas, se realizan mediciones, y se analizan los resultados de tiempo.
