METOD := Metodo
RUNGE := RungeKutta
RUN_I := RungeKutta_i
BASHF := AdamsBashford
BAS_I := AdamsBashford_i
MOULT := AdamsMoulton
MOU_I := AdamsMoulton_i

PROBP := Problema
PROB0 := prueba
PROB1 := simpleadvdiff1d
PROB2 := advdiff1d
PROB3 := brusselator1d
PROB4 := brusselator2d

DIR_H := Headers
DIR_O := Obj
DIR_S := Source
DIR_D := Datos

ALL_HEADERS := $(DIR_H)/$(METOD).h   $(DIR_H)/$(PROBP).h   $(DIR_H)/$(RUNGE).h   $(DIR_H)/$(RUN_I).h \
			   $(DIR_H)/$(BASHF).h   $(DIR_H)/$(BAS_I).h   $(DIR_H)/$(MOULT).h   $(DIR_H)/$(MOU_I).h \
               $(DIR_H)/$(PROB1).h   $(DIR_H)/$(PROB2).h   $(DIR_H)/$(PROB3).h   $(DIR_H)/$(PROB4).h   #$(DIR_H)/$(PROB0).h 

ALL_OBJ     := $(DIR_O)/$(METOD).o   $(DIR_O)/$(PROBP).o   $(DIR_O)/$(RUNGE).o   $(DIR_O)/$(RUN_I).o \
			   $(DIR_O)/$(BASHF).o   $(DIR_O)/$(BAS_I).o   $(DIR_O)/$(MOULT).o   $(DIR_O)/$(MOU_I).o \
               $(DIR_O)/$(PROB1).o   $(DIR_O)/$(PROB2).o   $(DIR_O)/$(PROB3).o   $(DIR_O)/$(PROB4).o   #$(DIR_O)/$(PROB0).o 

ALL_SRC     := $(DIR_S)/$(METOD).cpp $(DIR_S)/$(PROBP).cpp $(DIR_S)/$(RUNGE).cpp $(DIR_S)/$(RUN_I).cpp \
			   $(DIR_S)/$(BASHF).cpp $(DIR_S)/$(BAS_I).cpp $(DIR_S)/$(MOULT).cpp $(DIR_S)/$(MOU_I).cpp \
	           $(DIR_S)/$(PROB1).cpp $(DIR_S)/$(PROB2).cpp $(DIR_S)/$(PROB3).cpp $(DIR_S)/$(PROB4).cpp #$(DIR_S)/$(PROB0).cpp 
			   
ALL_DATA    := $(DIR_D)/datos0.txt   $(DIR_D)/datos1.txt 

CC := g++

NUM_PROBS := 1 2 3 4
NUM_HEBRAS := 1 2 4 8 16
TAMS := 100 200 500 1000 2000 5000 10000

# Creacion del ejecutable
create: solverOMP

solverOMP: $(DIR_O)/main.o $(ALL_OBJ)
	$(CC) -I$(DIR_H) -fopenmp $^ -O3 -o $@

$(DIR_O)/main.o: $(DIR_S)/main.cpp
	$(CC) -fopenmp -I$(DIR_H) -c $< -O3 -o $@

$(ALL_OBJ): $(DIR_O)/%.o: $(DIR_S)/%.cpp $(DIR_H)/%.h 
	$(CC) -fopenmp -I$(DIR_H) -c $< -O3 -o $@

# Ejecucion de problema con las variables introducidas
run:
	./solverOMP $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) 

#exportar con un método u otro
exporta:
	-rm $(DIR_D)/*.png 
	cd ./Datos/; python3 visualiza1234.py datos0.txt
	cd ./Datos/; python3 visualiza1234.py datos1.txt


# create + run + exportacion de datos
completo: create run exporta

clean:
	-rm $(DIR_O)/*.o
	-rm ./solverOMP

mrproper: clean
	-rm $(DIR_D)/*.txt
	-rm $(DIR_D)/*.png