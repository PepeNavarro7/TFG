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
	./solverOMP $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) $(modo)

runs:
	@./solverOMP $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) $(modo)

NUM_METODO := 1 2 3 4 5 6
NUM_ORDEN := 1 2 3 4 5
NUM_PROBLEMA := 1 2 3 4
num_prob1 := 1 2 3
num_prob2 := 4
NUM_HEBRAS := 1 4 16
tam_vec1 := 20000 25000 30000 40000 50000
tam_vec2 := 225 250
SALTO := 5 6 7

batch1:
	@for m in $(NUM_METODO); do \
		for p in $(num_prob1); do \
			for t in $(tam_vec1); do \
				for h in $(NUM_HEBRAS); do \
					echo "Ejecutando: metodo=$$m orden=5/4 problema=$$p hebras=$$h tamvector=$$t salto=6"; \
					$(MAKE) --no-print-directory runs metodo=$$m orden=5 problema=$$p hebras=$$h tamvector=$$t t0=0.0 tf=1.0 salto=6 modo=1; \
				done; \
			done; \
		done; \
    done

batch2:
	@for m in $(NUM_METODO); do \
		for p in $(num_prob2); do \
			for t in $(tam_vec2); do \
				for h in $(NUM_HEBRAS); do \
					echo "Ejecutando: metodo=$$m orden=5/4 problema=$$p hebras=$$h tamvector=$$t salto=6"; \
					$(MAKE) --no-print-directory runs metodo=$$m orden=5 problema=$$p hebras=$$h tamvector=$$t t0=0.0 tf=1.0 salto=6 modo=1; \
				done; \
			done; \
		done; \
    done

background:
	@nohup $(MAKE) batch1 > background.log 2>&1 &

background_2:
	@nohup sh -c "$(MAKE) batch1 && $(MAKE) batch2" > background.log 2>&1 &

stop:
	@echo "Deteniendo procesos batch..."
	@pkill -f "make batch1" || true
	@pkill -f "make batch2" || true
	@pkill -f solverOMP || true
	@echo "Procesos detenidos."


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
	-rm ./background.log

mrproper: clean
	-rm $(DIR_D)/*.txt
	-rm $(DIR_D)/*.png